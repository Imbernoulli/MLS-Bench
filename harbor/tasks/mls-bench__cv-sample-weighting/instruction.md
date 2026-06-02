# MLS-Bench: cv-sample-weighting

# CV Sample Reweighting Strategy Design

## Research Question
Design a class-weighting strategy for class-imbalanced image classification that improves balanced test accuracy on long-tail distributed datasets, across different architectures and imbalance ratios, while keeping the dataset construction, sampler, model, optimizer, and evaluation metric fixed.

## Background
Real-world datasets often follow long-tail class distributions: a few "head" classes dominate while many "tail" classes have very few samples. Uniform cross-entropy biases the classifier toward frequent classes, degrading performance on rare ones. Class reweighting assigns per-class weights to the cross-entropy loss to counteract this imbalance. Representative formulations include:

- **Inverse frequency**: `w[c] = N / (C * n[c])`, directly compensating for class size.
- **Square-root inverse**: `w[c] ∝ 1 / sqrt(n[c])`, a smoother variant that under-weights extreme rare-class amplification.
- **Effective Number of Samples** (Cui et al., CVPR 2019, arXiv:1901.05555): models data overlap with `E_n = (1 - β^n) / (1 - β)` and uses `w[c] ∝ 1 / E_{n[c]}`; the paper reports `β ∈ {0.9, 0.99, 0.999, 0.9999}` with `β=0.9999` typical for long-tail CIFAR.
- **Balanced Softmax-style weighting** (Ren et al., "Balanced Meta-Softmax for Long-Tailed Visual Recognition", NeurIPS 2020, arXiv:2007.10740): rebalances the softmax via a prior derived from class frequencies; equivalent in our setting to a particular weighting form on the loss.
- **LDAM** (Cao et al., NeurIPS 2019, arXiv:1906.07413): a related label-distribution-aware margin formulation, often combined with deferred reweighting.

These methods define different mappings from class frequency to loss weight, and may behave differently across datasets and imbalance regimes.

## What You Can Modify
The `compute_class_weights(class_counts, num_classes, config)` function inside `pytorch-vision/custom_weighting.py`. The function receives per-class sample counts and must return a 1-D tensor of length `num_classes` suitable for `nn.CrossEntropyLoss(weight=...)`.

`config` provides:
- `imbalance_ratio` (float)
- `dataset` (str)
- `arch` (str)
- `total_samples` (int)

You may modify the functional form mapping class counts to weights (inverse, power-law, logarithmic, piecewise, effective-number, etc.), use any field from `config`, choose any normalization strategy (sum to `C`, sum to `1`, unnormalized), and combine multiple ideas. The computation must be pure: no access to training data, model parameters, or test labels.

## Fixed Pipeline
- Optimizer: SGD with `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`.
- Schedule: cosine annealing over `200` epochs.
- Data augmentation: `RandomCrop(32, pad=4)` + `RandomHorizontalFlip`.
- Evaluation is on the balanced test set; training is on the long-tail train split.

## Baselines
- **inverse_freq** — `w[c] = total_samples / (num_classes * n[c])`.
- **effective_number** — Cui et al., arXiv:1901.05555; default `β=0.9999` (paper-recommended for long-tail CIFAR-100), with weights normalized so they sum to `num_classes`.
- **balanced_softmax** — weighting form motivated by Ren et al., arXiv:2007.10740, derived from the empirical class prior.

## Implementation Contract
The weighting rule must produce numerically stable class weights compatible with cross-entropy and must not change the dataset construction, sampler, model architecture, optimizer, or evaluation metric.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/pytorch-vision/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `pytorch-vision/custom_weighting.py`
- editable lines **164–195**




## Readable Context


### `pytorch-vision/custom_weighting.py`  [EDITABLE — lines 164–195 only]

```python
     1: """CV Sample Reweighting Benchmark.
     2: 
     3: Train vision models (ResNet-32, VGG-16-BN) on long-tail imbalanced CIFAR
     4: to evaluate sample reweighting strategies for class-imbalanced classification.
     5: 
     6: FIXED: Model architectures, imbalanced dataset creation, data pipeline, training loop.
     7: EDITABLE: compute_class_weights() function.
     8: 
     9: Usage:
    10:     python custom_weighting.py --arch resnet32 --dataset cifar10 --imbalance-ratio 100 --seed 42
    11: """
    12: 
    13: import argparse
    14: import math
    15: import os
    16: import time
    17: 
    18: import numpy as np
    19: import torch
    20: import torch.nn as nn
    21: import torch.nn.functional as F
    22: import torch.optim as optim
    23: import torchvision
    24: import torchvision.transforms as transforms
    25: from torch.utils.data import DataLoader, Subset
    26: 
    27: 
    28: # ============================================================================
    29: # FIXED
    30: # ============================================================================
    31: 
    32: # ── Model Architectures ──
    33: 
    34: class BasicBlock(nn.Module):
    35:     """Basic residual block for CIFAR ResNets."""
    36:     expansion = 1
    37: 
    38:     def __init__(self, in_planes, planes, stride=1):
    39:         super().__init__()
    40:         self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
    41:         self.bn1 = nn.BatchNorm2d(planes)
    42:         self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
    43:         self.bn2 = nn.BatchNorm2d(planes)
    44:         self.shortcut = nn.Sequential()
    45:         if stride != 1 or in_planes != planes * self.expansion:
    46:             self.shortcut = nn.Sequential(
    47:                 nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
    48:                 nn.BatchNorm2d(planes * self.expansion),
    49:             )
    50: 
    51:     def forward(self, x):
    52:         out = F.relu(self.bn1(self.conv1(x)))
    53:         out = self.bn2(self.conv2(out))
    54:         out += self.shortcut(x)
    55:         return F.relu(out)
    56: 
    57: 
    58: class ResNet(nn.Module):
    59:     """CIFAR-adapted ResNet (He et al., 2016).
    60: 
    61:     Uses 3x3 initial conv (no 7x7), no max pooling, global avg pool at end.
    62:     ResNet-32: [5,5,5] blocks.
    63:     """
    64: 
    65:     def __init__(self, block, num_blocks, num_classes=10):
    66:         super().__init__()
    67:         self.in_planes = 16
    68:         self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
    69:         self.bn1 = nn.BatchNorm2d(16)
    70:         self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
    71:         self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
    72:         self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
    73:         self.fc = nn.Linear(64 * block.expansion, num_classes)
    74: 
    75:     def _make_layer(self, block, planes, num_blocks, stride):
    76:         strides = [stride] + [1] * (num_blocks - 1)
    77:         layers = []
    78:         for s in strides:
    79:             layers.append(block(self.in_planes, planes, s))
    80:             self.in_planes = planes * block.expansion
    81:         return nn.Sequential(*layers)
    82: 
    83:     def forward(self, x):
    84:         out = F.relu(self.bn1(self.conv1(x)))
    85:         out = self.layer1(out)
    86:         out = self.layer2(out)
    87:         out = self.layer3(out)
    88:         out = F.adaptive_avg_pool2d(out, 1)
    89:         out = out.view(out.size(0), -1)
    90:         return self.fc(out)
    91: 
    92: 
    93: class VGG(nn.Module):
    94:     """VGG-16 with BatchNorm, adapted for CIFAR (Simonyan & Zisserman, 2015).
    95: 
    96:     Uses adaptive avg pool instead of large FC layers, suitable for 32x32 input.
    97:     """
    98: 
    99:     VGG16_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
   100:                  512, 512, 512, 'M', 512, 512, 512, 'M']
   101: 
   102:     def __init__(self, num_classes=100):
   103:         super().__init__()
   104:         self.features = self._make_layers(self.VGG16_CFG)
   105:         self.classifier = nn.Sequential(
   106:             nn.Linear(512, 512),
   107:             nn.ReLU(True),
   108:             nn.Dropout(0.5),
   109:             nn.Linear(512, num_classes),
   110:         )
   111: 
   112:     def _make_layers(self, cfg):
   113:         layers = []
   114:         in_channels = 3
   115:         for v in cfg:
   116:             if v == 'M':
   117:                 layers.append(nn.MaxPool2d(2, 2))
   118:             else:
   119:                 layers += [
   120:                     nn.Conv2d(in_channels, v, 3, padding=1),
   121:                     nn.BatchNorm2d(v),
   122:                     nn.ReLU(inplace=True),
   123:                 ]
   124:                 in_channels = v
   125:         return nn.Sequential(*layers)
   126: 
   127:     def forward(self, x):
   128:         x = self.features(x)
   129:         x = F.adaptive_avg_pool2d(x, 1)
   130:         x = x.view(x.size(0), -1)
   131:         return self.classifier(x)
   132: 
   133: 
   134: def build_model(arch, num_classes):
   135:     """Build model by architecture name."""
   136:     if arch == 'resnet32':
   137:         return ResNet(BasicBlock, [5, 5, 5], num_classes)
   138:     elif arch == 'vgg16bn':
   139:         return VGG(num_classes)
   140:     else:
   141:         raise ValueError(f"Unknown architecture: {arch}")
   142: 
   143: 
   144: # ── Weight Initialization (standard Kaiming) ──
   145: 
   146: def initialize_weights(model):
   147:     """Standard Kaiming initialization."""
   148:     for m in model.modules():
   149:         if isinstance(m, nn.Conv2d):
   150:             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
   151:         elif isinstance(m, nn.BatchNorm2d):
   152:             nn.init.constant_(m.weight, 1)
   153:             nn.init.constant_(m.bias, 0)
   154:         elif isinstance(m, nn.Linear):
   155:             nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
   156:             if m.bias is not None:
   157:                 nn.init.constant_(m.bias, 0)
   158: 
   159: 
   160: # ============================================================================
   161: # EDITABLE
   162: # ============================================================================
   163: # -- EDITABLE REGION START (lines 164-195) ------------------------------------
   164: def compute_class_weights(class_counts, num_classes, config):
   165:     """Compute per-class loss weights for imbalanced classification.
   166: 
   167:     Called after creating the imbalanced dataset, before training begins.
   168:     The returned weights are used as: nn.CrossEntropyLoss(weight=weights).
   169: 
   170:     Args:
   171:         class_counts: torch.Tensor of shape [num_classes] — number of training
   172:             samples per class (sorted by class index, class 0 has the most samples).
   173:         num_classes: int — number of classes (10 for CIFAR-10, 100 for CIFAR-100).
   174:         config: dict with keys:
   175:             - imbalance_ratio: float (e.g. 100.0 or 50.0)
   176:             - dataset: str ('cifar10' or 'cifar100')
   177:             - arch: str ('resnet32' or 'vgg16bn')
   178:             - total_samples: int (total training samples after imbalancing)
   179: 
   180:     Returns:
   181:         torch.Tensor of shape [num_classes] — per-class weights for CrossEntropyLoss.
   182:             Higher weight = more emphasis on that class during training.
   183: 
   184:     Design considerations:
   185:         - The dataset follows exponential imbalance: class i has
   186:           n_max * (1/imbalance_ratio)^(i/(C-1)) samples.
   187:         - Class 0 (most frequent) may have 5000 samples while class C-1
   188:           (rarest) may have only 50 samples (for ratio=100).
   189:         - Simple uniform weights (no reweighting) tend to bias toward
   190:           frequent classes.
   191:         - Inverse frequency weighting can overfit to rare classes.
   192:         - The optimal strategy balances between these extremes.
   193:     """
   194:     # Default: uniform weights (no reweighting)
   195:     return torch.ones(num_classes)
   196: # -- EDITABLE REGION END (lines 164-195) --------------------------------------
   197: 
   198: # ============================================================================
   199: # FIXED
   200: # ============================================================================
   201: 
   202: # ── Imbalanced Dataset Creation ──
   203: 
   204: def create_imbalanced_cifar(dataset, imbalance_ratio, num_classes, seed=42):
   205:     """Create a long-tail imbalanced version of a CIFAR dataset.
   206: 
   207:     Uses exponential decay: class i gets n_i = n_max * (1/imbalance_ratio)^(i/(C-1))
   208:     samples, where n_max is the original per-class count.
   209: 
   210:     Args:
   211:         dataset: torchvision CIFAR dataset (full balanced training set).
   212:         imbalance_ratio: float — ratio between most and least frequent class.
   213:         num_classes: int.
   214: 
   215:     Returns:
   216:         imbalanced_dataset: Subset with imbalanced class distribution.
   217:         class_counts: torch.Tensor [num_classes] — samples per class.
   218:     """
   219:     targets = np.array(dataset.targets)
   220:     # Original per-class count (CIFAR-10: 5000, CIFAR-100: 500)
   221:     n_max = np.sum(targets == 0)
   222: 
   223:     # Compute per-class sample counts via exponential decay
   224:     class_counts_np = np.zeros(num_classes, dtype=np.int64)
   225:     for c in range(num_classes):
   226:         mu = (1.0 / imbalance_ratio) ** (c / (num_classes - 1))
   227:         class_counts_np[c] = max(int(n_max * mu), 1)
   228: 
   229:     # Select subset indices
   230:     selected_indices = []
   231:     rng = np.random.RandomState(seed)
   232:     for c in range(num_classes):
   233:         class_indices = np.where(targets == c)[0]
   234:         rng.shuffle(class_indices)
   235:         selected_indices.extend(class_indices[:class_counts_np[c]])
   236: 
   237:     imbalanced_dataset = Subset(dataset, selected_indices)
   238:     class_counts = torch.tensor(class_counts_np, dtype=torch.float32)
   239:     return imbalanced_dataset, class_counts
   240: 
   241: 
   242: # ── Data Loading ──
   243: 
   244: def get_dataloaders(dataset_name, data_root, imbalance_ratio, batch_size=128, num_workers=4, seed=42):
   245:     """Create imbalanced CIFAR train and balanced test dataloaders."""
   246:     if dataset_name == 'cifar10':
   247:         mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
   248:         num_classes = 10
   249:         Dataset = torchvision.datasets.CIFAR10
   250:     elif dataset_name == 'cifar100':
   251:         mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
   252:         num_classes = 100
   253:         Dataset = torchvision.datasets.CIFAR100
   254:     else:
   255:         raise ValueError(f"Unknown dataset: {dataset_name}")
   256: 
   257:     train_transform = transforms.Compose([
   258:         transforms.RandomCrop(32, padding=4),
   259:         transforms.RandomHorizontalFlip(),
   260:         transforms.ToTensor(),
   261:         transforms.Normalize(mean, std),
   262:     ])
   263:     test_transform = transforms.Compose([
   264:         transforms.ToTensor(),
   265:         transforms.Normalize(mean, std),
   266:     ])
   267: 
   268:     full_train_set = Dataset(root=data_root, train=True, download=False, transform=train_transform)
   269:     test_set = Dataset(root=data_root, train=False, download=False, transform=test_transform)
   270: 
   271:     # Create imbalanced training set
   272:     imbalanced_train, class_counts = create_imbalanced_cifar(
   273:         full_train_set, imbalance_ratio, num_classes, seed,
   274:     )
   275: 
   276:     train_loader = DataLoader(
   277:         imbalanced_train, batch_size=batch_size, shuffle=True,
   278:         num_workers=num_workers, pin_memory=True,
   279:     )
   280:     test_loader = DataLoader(
   281:         test_set, batch_size=batch_size, shuffle=False,
   282:         num_workers=num_workers, pin_memory=True,
   283:     )
   284:     return train_loader, test_loader, num_classes, class_counts
   285: 
   286: 
   287: # ── Training Loop ──
   288: 
   289: def train_epoch(model, loader, criterion, optimizer, device):
   290:     """Train for one epoch. Returns (avg_loss, accuracy%)."""
   291:     model.train()
   292:     total_loss, correct, total = 0.0, 0, 0
   293:     for inputs, targets in loader:
   294:         inputs, targets = inputs.to(device), targets.to(device)
   295:         optimizer.zero_grad()
   296:         outputs = model(inputs)
   297:         loss = criterion(outputs, targets)
   298:         loss.backward()
   299:         optimizer.step()
   300:         total_loss += loss.item() * inputs.size(0)
   301:         _, predicted = outputs.max(1)
   302:         correct += predicted.eq(targets).sum().item()
   303:         total += inputs.size(0)
   304:     return total_loss / total, 100.0 * correct / total
   305: 
   306: 
   307: def evaluate(model, loader, criterion, device):
   308:     """Evaluate on balanced test set. Returns (avg_loss, accuracy%)."""
   309:     model.eval()
   310:     total_loss, correct, total = 0.0, 0, 0
   311:     with torch.no_grad():
   312:         for inputs, targets in loader:
   313:             inputs, targets = inputs.to(device), targets.to(device)
   314:             outputs = model(inputs)
   315:             loss = criterion(outputs, targets)
   316:             total_loss += loss.item() * inputs.size(0)
   317:             _, predicted = outputs.max(1)
   318:             correct += predicted.eq(targets).sum().item()
   319:             total += inputs.size(0)
   320:     return total_loss / total, 100.0 * correct / total
   321: 
   322: 
   323: def main():
   324:     parser = argparse.ArgumentParser(description="CV Sample Reweighting Benchmark")
   325:     parser.add_argument('--arch', type=str, required=True,
   326:                         choices=['resnet32', 'vgg16bn'])
   327:     parser.add_argument('--dataset', type=str, required=True,
   328:                         choices=['cifar10', 'cifar100'])
   329:     parser.add_argument('--imbalance-ratio', type=float, required=True,
   330:                         help='Imbalance ratio between most and least frequent class')
   331:     parser.add_argument('--data-root', type=str, default='/data/cifar')
   332:     parser.add_argument('--epochs', type=int, default=200)
   333:     parser.add_argument('--batch-size', type=int, default=128)
   334:     parser.add_argument('--lr', type=float, default=0.1)
   335:     parser.add_argument('--momentum', type=float, default=0.9)
   336:     parser.add_argument('--weight-decay', type=float, default=5e-4)
   337:     parser.add_argument('--seed', type=int, default=42)
   338:     parser.add_argument('--output-dir', type=str, default='.')
   339:     args = parser.parse_args()
   340: 
   341:     torch.manual_seed(args.seed)
   342:     torch.cuda.manual_seed_all(args.seed)
   343:     np.random.seed(args.seed)
   344:     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   345: 
   346:     # Data
   347:     train_loader, test_loader, num_classes, class_counts = get_dataloaders(
   348:         args.dataset, args.data_root, args.imbalance_ratio, args.batch_size, seed=args.seed,
   349:     )
   350: 
   351:     total_samples = int(class_counts.sum().item())
   352:     print(f"Dataset: {args.dataset} (long-tail, imbalance_ratio={args.imbalance_ratio})", flush=True)
   353:     print(f"Total training samples: {total_samples} (balanced would be "
   354:           f"{num_classes * int(class_counts[0].item())})", flush=True)
   355:     print(f"Class counts — max: {int(class_counts[0].item())}, "
   356:           f"min: {int(class_counts[-1].item())}", flush=True)
   357: 
   358:     # Model
   359:     model = build_model(args.arch, num_classes)
   360:     initialize_weights(model)
   361: 
   362:     # Compute class weights
   363:     config = {
   364:         'imbalance_ratio': args.imbalance_ratio,
   365:         'dataset': args.dataset,
   366:         'arch': args.arch,
   367:         'total_samples': total_samples,
   368:     }
   369:     weights = compute_class_weights(class_counts, num_classes, config)
   370:     weights = weights.to(device)
   371: 
   372:     model = model.to(device)
   373: 
   374:     # Optimizer
   375:     criterion = nn.CrossEntropyLoss(weight=weights)
   376:     optimizer = optim.SGD(
   377:         model.parameters(), lr=args.lr,
   378:         momentum=args.momentum, weight_decay=args.weight_decay,
   379:     )
   380:     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
   381: 
   382:     # Train
   383:     best_acc = 0.0
   384:     for epoch in range(args.epochs):
   385:         train_loss, train_acc = train_epoch(
   386:             model, train_loader, criterion, optimizer, device,
   387:         )
   388:         test_loss, test_acc = evaluate(model, test_loader, criterion, device)
   389:         scheduler.step()
   390: 
   391:         if (epoch + 1) % 10 == 0 or epoch == 0:
   392:             print(
   393:                 f"TRAIN_METRICS: epoch={epoch+1} train_loss={train_loss:.4f} "
   394:                 f"train_acc={train_acc:.2f} test_loss={test_loss:.4f} "
   395:                 f"test_acc={test_acc:.2f} lr={optimizer.param_groups[0]['lr']:.6f}",
   396:                 flush=True,
   397:             )
   398: 
   399:         if test_acc > best_acc:
   400:             best_acc = test_acc
   401: 
   402:     print(f"TEST_METRICS: test_acc={best_acc:.2f}", flush=True)
   403: 
   404: 
   405: if __name__ == '__main__':
   406:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `inverse_freq` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_weighting.py`:

```python
Lines 164–180:
   161: # EDITABLE
   162: # ============================================================================
   163: # -- EDITABLE REGION START (lines 164-195) ------------------------------------
   164: def compute_class_weights(class_counts, num_classes, config):
   165:     """Inverse frequency weighting.
   166: 
   167:     weight[c] = total_samples / (num_classes * count[c]).
   168:     Directly proportional to inverse class frequency.
   169:     Smoothed via square-root dampening to prevent training instability
   170:     on architectures without skip connections (e.g. VGG).
   171:     Normalized so weights sum to num_classes.
   172:     """
   173:     total = class_counts.sum().float()
   174:     weights = total / (num_classes * class_counts.float())
   175:     # Square-root dampening: reduces dynamic range while preserving ordering
   176:     # For ratio=100 CIFAR-100: raw ratio ~100x -> dampened ~10x
   177:     weights = torch.sqrt(weights)
   178:     # Normalize so weights sum to num_classes
   179:     weights = weights / weights.sum() * num_classes
   180:     return weights
   181: # -- EDITABLE REGION END (lines 164-195) --------------------------------------
   182: 
   183: # ============================================================================
```

### `effective_number` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_weighting.py`:

```python
Lines 164–180:
   161: # EDITABLE
   162: # ============================================================================
   163: # -- EDITABLE REGION START (lines 164-195) ------------------------------------
   164: def compute_class_weights(class_counts, num_classes, config):
   165:     """Effective number of samples weighting (Cui et al., CVPR 2019).
   166: 
   167:     E_n = (1 - beta^n) / (1 - beta).
   168:     weight[c] = (1 - beta) / (1 - beta^count[c]).
   169:     Uses beta=0.9999, a task-local value explored in class-balanced losses.
   170:     Smoothed via square-root dampening to prevent training instability
   171:     on architectures without skip connections (e.g. VGG).
   172:     """
   173:     beta = 0.9999
   174:     effective_num = 1.0 - torch.pow(beta, class_counts.float())
   175:     weights = (1.0 - beta) / effective_num
   176:     # Square-root dampening: reduces dynamic range while preserving ordering
   177:     weights = torch.sqrt(weights)
   178:     # Normalize so weights sum to num_classes
   179:     weights = weights / weights.sum() * num_classes
   180:     return weights
   181: # -- EDITABLE REGION END (lines 164-195) --------------------------------------
   182: 
   183: # ============================================================================
```

### `balanced_softmax` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_weighting.py`:

```python
Lines 164–186:
   161: # EDITABLE
   162: # ============================================================================
   163: # -- EDITABLE REGION START (lines 164-195) ------------------------------------
   164: def compute_class_weights(class_counts, num_classes, config):
   165:     """Balanced Softmax-inspired log-prior weighting (Ren et al., NeurIPS 2020).
   166: 
   167:     Balanced Softmax adjusts logits by subtracting log(pi_c). In this task's
   168:     class-weighting interface, we use weights derived from the log-frequency gap:
   169:     weight[c] = 1 + alpha * (log(n_max) - log(n_c)), alpha chosen so that
   170:     the max/min weight ratio is moderate (~5:1).
   171:     This provides a log-scale reweighting that is gentler than inverse
   172:     frequency but still informed by class prior.
   173:     """
   174:     log_counts = torch.log(class_counts.float())
   175:     log_max = log_counts.max()
   176:     # Log-gap weights: classes further from max count get higher weight
   177:     gap = log_max - log_counts  # 0 for most frequent, log(ratio) for rarest
   178:     # Scale so the rarest class gets weight ~5x the most frequent
   179:     max_gap = gap.max()
   180:     if max_gap > 0:
   181:         weights = 1.0 + 4.0 * (gap / max_gap)
   182:     else:
   183:         weights = torch.ones(num_classes)
   184:     # Normalize so weights sum to num_classes
   185:     weights = weights / weights.sum() * num_classes
   186:     return weights
   187: # -- EDITABLE REGION END (lines 164-195) --------------------------------------
   188: 
   189: # ============================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
