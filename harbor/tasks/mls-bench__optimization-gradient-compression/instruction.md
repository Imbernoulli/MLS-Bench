# MLS-Bench: optimization-gradient-compression

# Gradient Compression for Communication-Efficient Distributed Training

## Research Question
Design a gradient compression operator that reduces communication cost in distributed training while maintaining convergence quality (test accuracy).

## Background
In distributed data-parallel training, gradient communication is often the bottleneck. Workers compute local gradients, which must be aggregated (e.g., via all-reduce) before the optimizer step. Gradient compression reduces the volume of data communicated by applying lossy compression to gradients before transmission.

Three main families of compression exist:
- **Sparsification**: keep only a subset of gradient elements (e.g., TopK selects the largest magnitudes; Stich, Cordonnier, and Jaggi, "Sparsified SGD with Memory", NeurIPS 2018).
- **Quantization**: reduce the precision of gradient values (e.g., QSGD uses stochastic rounding to discrete levels).
- **Low-rank approximation**: approximate gradient matrices with low-rank factors (e.g., PowerSGD).

A key challenge is that naive compression introduces bias or variance that degrades convergence. Error feedback — accumulating compression residuals locally and adding them to the next gradient — is a widely used correction (Karimireddy, Rebjock, Stich, and Jaggi, "Error Feedback Fixes SignSGD and Other Gradient Compression Schemes", ICML 2019; arXiv:1901.09847).

## Task
Modify the `Compressor` class in `custom_compressor.py`. Your compressor must implement:
- `__init__(self, compress_ratio)`: initialize with a target compression ratio (`0.01` = 100x compression).
- `compress(self, tensor, name)`: compress a gradient tensor, returning `(compressed_tensors, ctx)`.
- `decompress(self, compressed_tensors, ctx)`: reconstruct the gradient.

The compressor may maintain internal state (e.g., error feedback residuals) across calls. The `name` parameter identifies parameters for per-parameter state tracking.

## Interface
```python
class Compressor:
    def __init__(self, compress_ratio=0.01): ...
    def compress(self, tensor, name) -> (list[Tensor], ctx): ...
    def decompress(self, compressed_tensors, ctx) -> Tensor: ...
```
- `compress_ratio`: fraction of gradient elements/information to retain (`0.01` = keep 1%).
- `compressed_tensors`: list of tensors that would be communicated over the network.
- `ctx`: local context (not communicated) needed for decompression.
- The decompressed tensor must have the same shape as the original input.

## Evaluation
Trained and evaluated on three settings with 100x compression (`compress_ratio = 0.01`):
- **ResNet-20 / CIFAR-10** (~0.27M params): small model, standard benchmark.
- **VGG-11-BN / CIFAR-100** (~9.8M params): larger model, harder 100-class problem.
- **ResNet-56 / CIFAR-10** (~0.85M params): deeper model, tests scalability.

Metric: **best test accuracy** (higher is better). All settings use SGD with momentum, cosine LR schedule, and 200 training epochs.

## Baselines (paper-cited reference implementations)
- **topk_ef** — Top-K sparsification with error feedback (Stich et al., "Sparsified SGD with Memory", NeurIPS 2018; Karimireddy et al., "Error Feedback Fixes SignSGD and Other Gradient Compression Schemes", ICML 2019; arXiv:1901.09847). Keeps the `k = compress_ratio * d` largest-magnitude entries.
- **qsgd** — Quantized SGD with stochastic uniform quantization (Alistarh, Grubic, Li, Tomioka, and Vojnovic, "QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding", NeurIPS 2017; arXiv:1610.02132).
- **signsgd** — Sign-only gradient compression (Bernstein, Wang, Azizzadenesheli, and Anandkumar, "signSGD: Compressed Optimisation for Non-Convex Problems", ICML 2018; arXiv:1802.04434), typically combined with majority-vote aggregation.

A reference low-rank method (Vogels, Karimireddy, and Jaggi, "PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization", NeurIPS 2019; arXiv:1905.13727) is a useful design point even though it is not run as a baseline here.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/pytorch-vision/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `pytorch-vision/custom_compressor.py`
- editable lines **182–232**




## Readable Context


### `pytorch-vision/custom_compressor.py`  [EDITABLE — lines 182–232 only]

```python
     1: """Gradient Compression for Communication-Efficient Distributed Training.
     2: 
     3: Self-contained benchmark: trains standard vision models on CIFAR datasets
     4: using data-parallel SGD with a pluggable gradient compressor.
     5: 
     6: The script simulates distributed training on a single node by:
     7: 1. Computing gradients normally
     8: 2. Applying compress() -> decompress() to each gradient (simulating communication)
     9: 3. Using the decompressed gradient for the optimizer step
    10: 
    11: This faithfully measures the effect of gradient compression on convergence
    12: quality, which is the core ML-science question, without requiring multi-node
    13: infrastructure.
    14: """
    15: 
    16: import argparse
    17: import math
    18: import os
    19: import time
    20: 
    21: import torch
    22: import torch.nn as nn
    23: import torch.nn.functional as F
    24: import torch.optim as optim
    25: from torch.utils.data import DataLoader
    26: from torchvision import datasets, transforms
    27: 
    28: # ============================================================================
    29: # Model Definitions (FIXED)
    30: # ============================================================================
    31: 
    32: 
    33: def conv3x3(in_planes, out_planes, stride=1):
    34:     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    35:                      padding=1, bias=False)
    36: 
    37: 
    38: class BasicBlock(nn.Module):
    39:     expansion = 1
    40: 
    41:     def __init__(self, in_planes, planes, stride=1):
    42:         super().__init__()
    43:         self.conv1 = conv3x3(in_planes, planes, stride)
    44:         self.bn1 = nn.BatchNorm2d(planes)
    45:         self.conv2 = conv3x3(planes, planes)
    46:         self.bn2 = nn.BatchNorm2d(planes)
    47:         self.shortcut = nn.Sequential()
    48:         if stride != 1 or in_planes != planes * self.expansion:
    49:             self.shortcut = nn.Sequential(
    50:                 nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1,
    51:                           stride=stride, bias=False),
    52:                 nn.BatchNorm2d(planes * self.expansion),
    53:             )
    54: 
    55:     def forward(self, x):
    56:         out = F.relu(self.bn1(self.conv1(x)))
    57:         out = self.bn2(self.conv2(out))
    58:         out += self.shortcut(x)
    59:         return F.relu(out)
    60: 
    61: 
    62: class ResNet(nn.Module):
    63:     def __init__(self, block, num_blocks, num_classes=10):
    64:         super().__init__()
    65:         self.in_planes = 16
    66:         self.conv1 = conv3x3(3, 16)
    67:         self.bn1 = nn.BatchNorm2d(16)
    68:         self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
    69:         self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
    70:         self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
    71:         self.linear = nn.Linear(64 * block.expansion, num_classes)
    72: 
    73:     def _make_layer(self, block, planes, num_blocks, stride):
    74:         strides = [stride] + [1] * (num_blocks - 1)
    75:         layers = []
    76:         for s in strides:
    77:             layers.append(block(self.in_planes, planes, s))
    78:             self.in_planes = planes * block.expansion
    79:         return nn.Sequential(*layers)
    80: 
    81:     def forward(self, x):
    82:         out = F.relu(self.bn1(self.conv1(x)))
    83:         out = self.layer1(out)
    84:         out = self.layer2(out)
    85:         out = self.layer3(out)
    86:         out = F.adaptive_avg_pool2d(out, 1)
    87:         out = out.view(out.size(0), -1)
    88:         return self.linear(out)
    89: 
    90: 
    91: class VGG(nn.Module):
    92:     """VGG-11 with batch normalization."""
    93: 
    94:     def __init__(self, num_classes=100):
    95:         super().__init__()
    96:         cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    97:         layers = []
    98:         in_channels = 3
    99:         for v in cfg:
   100:             if v == 'M':
   101:                 layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
   102:             else:
   103:                 layers.extend([
   104:                     nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
   105:                     nn.BatchNorm2d(v),
   106:                     nn.ReLU(inplace=True),
   107:                 ])
   108:                 in_channels = v
   109:         self.features = nn.Sequential(*layers)
   110:         self.classifier = nn.Sequential(
   111:             nn.Linear(512, 512),
   112:             nn.ReLU(True),
   113:             nn.Dropout(),
   114:             nn.Linear(512, 512),
   115:             nn.ReLU(True),
   116:             nn.Dropout(),
   117:             nn.Linear(512, num_classes),
   118:         )
   119: 
   120:     def forward(self, x):
   121:         x = self.features(x)
   122:         x = F.adaptive_avg_pool2d(x, 1)
   123:         x = x.view(x.size(0), -1)
   124:         return self.classifier(x)
   125: 
   126: 
   127: def build_model(model_name, num_classes, device):
   128:     if model_name == 'resnet20':
   129:         model = ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)
   130:     elif model_name == 'resnet56':
   131:         model = ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)
   132:     elif model_name == 'vgg11':
   133:         model = VGG(num_classes=num_classes)
   134:     else:
   135:         raise ValueError(f"Unknown model: {model_name}")
   136:     return model.to(device)
   137: 
   138: 
   139: # ============================================================================
   140: # Data Loading (FIXED)
   141: # ============================================================================
   142: 
   143: def get_dataloaders(dataset_name, batch_size, num_workers=2):
   144:     if dataset_name == 'cifar10':
   145:         mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
   146:         num_classes = 10
   147:         Dataset = datasets.CIFAR10
   148:     elif dataset_name == 'cifar100':
   149:         mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
   150:         num_classes = 100
   151:         Dataset = datasets.CIFAR100
   152:     else:
   153:         raise ValueError(f"Unknown dataset: {dataset_name}")
   154: 
   155:     train_transform = transforms.Compose([
   156:         transforms.RandomCrop(32, padding=4),
   157:         transforms.RandomHorizontalFlip(),
   158:         transforms.ToTensor(),
   159:         transforms.Normalize(mean, std),
   160:     ])
   161:     test_transform = transforms.Compose([
   162:         transforms.ToTensor(),
   163:         transforms.Normalize(mean, std),
   164:     ])
   165: 
   166:     _data_root = os.environ.get("DATA_ROOT", "/data")
   167:     train_set = Dataset(_data_root + '/cifar', train=True, download=False,
   168:                         transform=train_transform)
   169:     test_set = Dataset(_data_root + '/cifar', train=False, download=False,
   170:                        transform=test_transform)
   171: 
   172:     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
   173:                               num_workers=num_workers, pin_memory=True)
   174:     test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
   175:                              num_workers=num_workers, pin_memory=True)
   176:     return train_loader, test_loader, num_classes
   177: 
   178: 
   179: # ============================================================================
   180: # EDITABLE SECTION — Gradient Compressor (lines 182-232)
   181: # ============================================================================
   182: 
   183: class Compressor:
   184:     """Gradient compressor base implementation.
   185: 
   186:     Interface contract:
   187:     - compress(tensor) -> (compressed_tensors: list[Tensor], ctx: any)
   188:         Compress a gradient tensor. Only `compressed_tensors` would be
   189:         "communicated" in a real distributed setting. `ctx` stays local.
   190:     - decompress(compressed_tensors, ctx) -> Tensor
   191:         Reconstruct the gradient from compressed representation.
   192:         Must return a tensor of the same shape as the original.
   193:     - The compressor may maintain internal state (e.g., error feedback
   194:         residuals) across calls for the same parameter.
   195: 
   196:     Default: identity (no compression). Replace with your method.
   197:     """
   198: 
   199:     def __init__(self, compress_ratio=0.01):
   200:         """Initialize the compressor.
   201: 
   202:         Args:
   203:             compress_ratio: Target compression ratio (fraction of elements
   204:                 to keep for sparsification, or quantization level).
   205:                 0.01 = 100x compression, 0.1 = 10x compression.
   206:         """
   207:         self.compress_ratio = compress_ratio
   208: 
   209:     def compress(self, tensor, name):
   210:         """Compress a gradient tensor.
   211: 
   212:         Args:
   213:             tensor: Gradient tensor to compress (flattened or original shape).
   214:             name: Parameter name (useful for maintaining per-parameter state).
   215: 
   216:         Returns:
   217:             compressed_tensors: list of tensors that would be communicated.
   218:             ctx: local context needed for decompression (not communicated).
   219:         """
   220:         return [tensor.clone()], tensor.shape
   221: 
   222:     def decompress(self, compressed_tensors, ctx):
   223:         """Decompress gradients back to original shape.
   224: 
   225:         Args:
   226:             compressed_tensors: list of tensors from compress().
   227:             ctx: local context from compress().
   228: 
   229:         Returns:
   230:             Decompressed gradient tensor matching original shape.
   231:         """
   232:         return compressed_tensors[0].view(ctx)
   233: 
   234: 
   235: # ============================================================================
   236: # FIXED SECTION — Training Loop
   237: # ============================================================================
   238: 
   239: def cosine_lr(optimizer, epoch, total_epochs, warmup_epochs, base_lr, min_lr=0.0):
   240:     """Cosine learning rate schedule with linear warmup."""
   241:     if epoch < warmup_epochs:
   242:         lr = base_lr * (epoch + 1) / (warmup_epochs + 1)
   243:     else:
   244:         progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
   245:         lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
   246:     for param_group in optimizer.param_groups:
   247:         param_group['lr'] = lr
   248:     return lr
   249: 
   250: 
   251: def apply_gradient_compression(model, compressor):
   252:     """Apply gradient compression to all model parameters.
   253: 
   254:     Simulates the compress -> communicate -> decompress pipeline of
   255:     distributed training. In a real system, only compressed_tensors
   256:     would be sent over the network.
   257:     """
   258:     for name, param in model.named_parameters():
   259:         if param.grad is None:
   260:             continue
   261:         grad = param.grad.data
   262:         compressed, ctx = compressor.compress(grad, name)
   263:         decompressed = compressor.decompress(compressed, ctx)
   264:         param.grad.data = decompressed
   265: 
   266: 
   267: def evaluate(model, test_loader, device):
   268:     model.eval()
   269:     correct = 0
   270:     total = 0
   271:     total_loss = 0.0
   272:     with torch.no_grad():
   273:         for images, labels in test_loader:
   274:             images, labels = images.to(device), labels.to(device)
   275:             outputs = model(images)
   276:             loss = F.cross_entropy(outputs, labels, reduction='sum')
   277:             total_loss += loss.item()
   278:             _, predicted = outputs.max(1)
   279:             total += labels.size(0)
   280:             correct += predicted.eq(labels).sum().item()
   281:     acc = 100.0 * correct / total
   282:     avg_loss = total_loss / total
   283:     return acc, avg_loss
   284: 
   285: 
   286: def train(args):
   287:     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   288:     torch.manual_seed(args.seed)
   289: 
   290:     train_loader, test_loader, num_classes = get_dataloaders(
   291:         args.dataset, args.batch_size)
   292:     model = build_model(args.model, num_classes, device)
   293: 
   294:     n_params = sum(p.numel() for p in model.parameters())
   295:     print(f"Model: {args.model}, Dataset: {args.dataset}, "
   296:           f"Parameters: {n_params:,}, Compress ratio: {args.compress_ratio}")
   297: 
   298:     optimizer = optim.SGD(model.parameters(), lr=args.lr,
   299:                           momentum=0.9, weight_decay=args.weight_decay)
   300: 
   301:     compressor = Compressor(compress_ratio=args.compress_ratio)
   302: 
   303:     best_acc = 0.0
   304:     for epoch in range(args.epochs):
   305:         lr = cosine_lr(optimizer, epoch, args.epochs, args.warmup_epochs,
   306:                        args.lr, min_lr=args.lr * 0.01)
   307:         model.train()
   308:         running_loss = 0.0
   309:         correct = 0
   310:         total = 0
   311: 
   312:         for batch_idx, (images, labels) in enumerate(train_loader):
   313:             images, labels = images.to(device), labels.to(device)
   314: 
   315:             optimizer.zero_grad()
   316:             outputs = model(images)
   317:             loss = F.cross_entropy(outputs, labels)
   318:             loss.backward()
   319: 
   320:             # Apply gradient compression before optimizer step
   321:             apply_gradient_compression(model, compressor)
   322: 
   323:             optimizer.step()
   324: 
   325:             running_loss += loss.item()
   326:             _, predicted = outputs.max(1)
   327:             total += labels.size(0)
   328:             correct += predicted.eq(labels).sum().item()
   329: 
   330:         train_acc = 100.0 * correct / total
   331:         train_loss = running_loss / len(train_loader)
   332: 
   333:         if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
   334:             test_acc, test_loss = evaluate(model, test_loader, device)
   335:             if test_acc > best_acc:
   336:                 best_acc = test_acc
   337:             print(f"TRAIN_METRICS epoch={epoch+1} lr={lr:.6f} "
   338:                   f"train_loss={train_loss:.4f} train_acc={train_acc:.2f} "
   339:                   f"test_acc={test_acc:.2f} test_loss={test_loss:.4f}",
   340:                   flush=True)
   341:         else:
   342:             print(f"TRAIN_METRICS epoch={epoch+1} lr={lr:.6f} "
   343:                   f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}",
   344:                   flush=True)
   345: 
   346:     # Final evaluation
   347:     test_acc, test_loss = evaluate(model, test_loader, device)
   348:     if test_acc > best_acc:
   349:         best_acc = test_acc
   350:     print(f"TEST_METRICS test_acc={test_acc:.2f} best_acc={best_acc:.2f} "
   351:           f"test_loss={test_loss:.4f}", flush=True)
   352: 
   353: 
   354: def main():
   355:     parser = argparse.ArgumentParser(description='Gradient Compression Benchmark')
   356:     parser.add_argument('--model', type=str, default='resnet20',
   357:                         choices=['resnet20', 'resnet56', 'vgg11'])
   358:     parser.add_argument('--dataset', type=str, default='cifar10',
   359:                         choices=['cifar10', 'cifar100'])
   360:     parser.add_argument('--batch-size', type=int, default=128)
   361:     parser.add_argument('--epochs', type=int, default=200)
   362:     parser.add_argument('--lr', type=float, default=0.1)
   363:     parser.add_argument('--weight-decay', type=float, default=5e-4)
   364:     parser.add_argument('--warmup-epochs', type=int, default=5)
   365:     parser.add_argument('--compress-ratio', type=float, default=0.01,
   366:                         help='Compression ratio (fraction of gradient to keep)')
   367:     parser.add_argument('--seed', type=int, default=42)
   368:     args = parser.parse_args()
   369:     train(args)
   370: 
   371: 
   372: if __name__ == '__main__':
   373:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `topk_ef` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_compressor.py`:

```python
Lines 182–222:
   179: # ============================================================================
   180: # EDITABLE SECTION — Gradient Compressor (lines 182-232)
   181: # ============================================================================
   182: class Compressor:
   183:     """TopK sparsification with error feedback (EF-TopK).
   184: 
   185:     Keeps the K largest-magnitude gradient elements per tensor.
   186:     Error feedback accumulates the compression error (original - decompressed)
   187:     and adds it to the next gradient before compression, ensuring convergence.
   188:     """
   189: 
   190:     def __init__(self, compress_ratio=0.01):
   191:         self.compress_ratio = compress_ratio
   192:         self.residuals = {}
   193: 
   194:     def compress(self, tensor, name):
   195:         # Error feedback: add accumulated residual
   196:         if name in self.residuals:
   197:             tensor = tensor + self.residuals[name]
   198: 
   199:         shape = tensor.shape
   200:         tensor_flat = tensor.flatten()
   201:         numel = tensor_flat.numel()
   202:         k = max(1, int(numel * self.compress_ratio))
   203: 
   204:         # Select top-k by magnitude
   205:         _, indices = torch.topk(tensor_flat.abs(), k, sorted=False)
   206:         values = tensor_flat[indices]
   207: 
   208:         # Update residual: store what was NOT communicated
   209:         decompressed_flat = torch.zeros_like(tensor_flat)
   210:         decompressed_flat.scatter_(0, indices, values)
   211:         self.residuals[name] = tensor_flat - decompressed_flat
   212:         self.residuals[name] = self.residuals[name].view(shape)
   213: 
   214:         return [values, indices], (numel, shape)
   215: 
   216:     def decompress(self, compressed_tensors, ctx):
   217:         values, indices = compressed_tensors
   218:         numel, shape = ctx
   219:         tensor_decompressed = torch.zeros(
   220:             numel, dtype=values.dtype, device=values.device)
   221:         tensor_decompressed.scatter_(0, indices, values)
   222:         return tensor_decompressed.view(shape)
   223: 
   224: 
   225: # ============================================================================
```

### `qsgd` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_compressor.py`:

```python
Lines 182–252:
   179: # ============================================================================
   180: # EDITABLE SECTION — Gradient Compressor (lines 182-232)
   181: # ============================================================================
   182: class Compressor:
   183:     """QSGD — Quantized Stochastic Gradient Descent.
   184: 
   185:     Quantizes each gradient element to one of `s` discrete levels using
   186:     randomized rounding. The quantization is unbiased: E[Q(g)] = g.
   187:     Communication cost: O(n * log(s) / 32) of original, where n = numel.
   188: 
   189:     Uses s=256 quantization levels for a stable communication/variance tradeoff.
   190: 
   191:     Note: QSGD is an *unbiased* compressor, so error feedback is not needed
   192:     and can actually hurt convergence. Unlike biased compressors (TopK,
   193:     SignSGD) that systematically lose information, QSGD preserves the
   194:     expected gradient value, making the vanilla SGD convergence guarantees
   195:     applicable with only increased variance.
   196: 
   197:     Reference: Alistarh et al., "QSGD: Communication-Efficient SGD via
   198:     Gradient Quantization and Encoding", NeurIPS 2017.
   199:     """
   200: 
   201:     def __init__(self, compress_ratio=0.01):
   202:         self.compress_ratio = compress_ratio
   203:         # QSGD: s = number of quantization levels (~log2(s)+1 bits/element).
   204:         # Var(Q(g)) ~ ||g||^2 * min(d/s^2, sqrt(d)/s). For deep nets with
   205:         # d ~ 1e7 params, small s produces huge variance that interacts with
   206:         # momentum SGD causing divergence on some seeds. s=256 gives ~9
   207:         # bits/element (~3.5x compression) and keeps variance bounded.
   208:         self.quantum_num = 256
   209:         # Per-tensor gradient clip: prevent rare large-norm gradients from
   210:         # amplifying quantization noise into divergence (standard QSGD
   211:         # practice, cf. Alistarh 2017 Algorithm 1 discussion).
   212:         self.clip_norm = 1.0
   213: 
   214:     def compress(self, tensor, name):
   215:         shape = tensor.shape
   216:         tensor_flat = tensor.flatten()
   217: 
   218:         # Gradient clipping BEFORE quantization — critical for stability.
   219:         norm = tensor_flat.norm()
   220:         if norm == 0:
   221:             return [tensor_flat.to(torch.int16), norm], shape
   222:         clip_coef = self.clip_norm / (norm + 1e-6)
   223:         if clip_coef < 1.0:
   224:             tensor_flat = tensor_flat * clip_coef
   225:             norm = tensor_flat.norm()
   226:             if norm == 0:
   227:                 return [tensor_flat.to(torch.int16), norm], shape
   228: 
   229:         abs_gradient = tensor_flat.abs()
   230: 
   231:         # Quantize: level = floor(s * |g_i| / ||g||) with stochastic rounding
   232:         level_float = self.quantum_num / norm * abs_gradient
   233:         previous_level = level_float.floor()
   234:         prob = torch.rand_like(tensor_flat)
   235:         is_next_level = (prob < (level_float - previous_level)).float()
   236:         new_level = previous_level + is_next_level
   237: 
   238:         # Store sign and quantized level
   239:         sign = tensor_flat.sign()
   240:         tensor_compressed = (new_level * sign)
   241:         tensor_compressed = tensor_compressed.to(torch.int16)
   242: 
   243:         return [tensor_compressed, norm], shape
   244: 
   245:     def decompress(self, compressed_tensors, ctx):
   246:         shape = ctx
   247:         tensor_compressed, norm = compressed_tensors
   248: 
   249:         # Dequantize: g_hat = (norm / s) * quantized_value
   250:         decode_output = tensor_compressed.float()
   251:         tensor_decompressed = norm / self.quantum_num * decode_output
   252:         return tensor_decompressed.view(shape)
   253: 
   254: 
   255: # ============================================================================
```

### `signsgd` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_compressor.py`:

```python
Lines 182–227:
   179: # ============================================================================
   180: # EDITABLE SECTION — Gradient Compressor (lines 182-232)
   181: # ============================================================================
   182: class Compressor:
   183:     """SignSGD with error feedback.
   184: 
   185:     Compresses each gradient element to its sign (+1 or -1), achieving
   186:     32x compression. Error feedback accumulates the magnitude information
   187:     lost during sign extraction, improving convergence.
   188: 
   189:     The compress_ratio parameter is not used for sign compression (always
   190:     1-bit), but the error feedback momentum can be tuned.
   191:     """
   192: 
   193:     def __init__(self, compress_ratio=0.01):
   194:         self.compress_ratio = compress_ratio
   195:         self.residuals = {}
   196:         # Error feedback momentum
   197:         self.ef_beta = 1.0
   198: 
   199:     def compress(self, tensor, name):
   200:         # Error feedback: add accumulated residual
   201:         if name in self.residuals:
   202:             tensor = tensor + self.ef_beta * self.residuals[name]
   203: 
   204:         shape = tensor.shape
   205:         tensor_flat = tensor.flatten()
   206: 
   207:         # Sign compression: 1 bit per element
   208:         signs = (tensor_flat >= 0).to(torch.uint8)
   209: 
   210:         # Scale by mean magnitude for better reconstruction
   211:         mean_magnitude = tensor_flat.abs().mean()
   212: 
   213:         # Update residual: original - reconstructed
   214:         sign_float = signs.float() * 2 - 1  # map {0,1} -> {-1,+1}
   215:         reconstructed = sign_float * mean_magnitude
   216:         self.residuals[name] = (tensor_flat - reconstructed).view(shape)
   217: 
   218:         return [signs, mean_magnitude], shape
   219: 
   220:     def decompress(self, compressed_tensors, ctx):
   221:         shape = ctx
   222:         signs, mean_magnitude = compressed_tensors
   223: 
   224:         # Reconstruct: sign * mean_magnitude
   225:         sign_float = signs.float() * 2 - 1
   226:         tensor_decompressed = sign_float * mean_magnitude
   227:         return tensor_decompressed.view(shape)
   228: 
   229: 
   230: # ============================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
