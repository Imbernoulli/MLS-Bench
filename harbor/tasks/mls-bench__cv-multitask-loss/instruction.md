# MLS-Bench: cv-multitask-loss

# CV Multi-Task Loss Combination Strategy Design

## Research Question
Design a multi-task loss combination strategy for jointly training fine-grained and coarse classification on an image dataset with a hierarchical label structure, with the primary objective of maximizing fine-class test accuracy.

## Background
The training setup involves a visual recognition dataset with fine classes organized into coarse superclasses. Training a model with two classification heads (fine + coarse) provides a natural multi-task learning setup where the coarse task acts as an auxiliary signal. The key challenge is how to combine the two losses so the auxiliary signal helps rather than hurts the primary objective. Representative approaches include:

- **Equal weighting**: simply sum the per-task losses (the trivial baseline).
- **Uncertainty weighting** (Kendall, Gal & Cipolla, CVPR 2018, arXiv:1705.07115): learn a per-task log-variance `s_i` and combine losses as `sum_i (exp(-s_i) * L_i + s_i)`.
- **Dynamic Weight Average (DWA)** (Liu, Johns & Davison, "End-to-End Multi-Task Learning with Attention", CVPR 2019, arXiv:1803.10704): weight each task by the relative rate of change of its loss across recent epochs, with a temperature parameter (`T=2.0` is the value used in the paper).
- **PCGrad** (Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020, arXiv:2001.06782): when two task gradients have negative cosine similarity, project each onto the normal plane of the other to reduce gradient interference; otherwise leave them unchanged.
- **Random Loss Weighting**: simple stochastic weighting baseline used as a sanity check in some MTL studies.

The coarse labels encode semantic hierarchy, and balancing this auxiliary signal against the fine-class objective interacts non-trivially with architecture, training stage, and gradient geometry.

## What You Can Modify
The `MultiTaskLoss` class inside `pytorch-vision/custom_mtl.py`. The class receives the individual task losses and returns a single scalar loss.

The `forward` method receives:
- `fine_loss` (scalar tensor): cross-entropy for the 100-class fine head.
- `coarse_loss` (scalar tensor): cross-entropy for the 20-class coarse head.
- `epoch` (int): current epoch (0-indexed).
- `total_epochs` (int): total number of training epochs.

You may modify `__init__` to add learnable parameters (log-variances, weights, etc.), implement any combination strategy in `forward`, use `epoch` / `total_epochs` for curriculum or scheduling, and maintain auxiliary state such as loss-history buffers. The `MultiTaskLoss` parameters are included in the optimizer, so any registered learnable tensors will be trained.

## Fixed Pipeline
- Optimizer: SGD with `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`.
- Schedule: cosine annealing over `200` epochs.
- Data augmentation: `RandomCrop(32, pad=4)` + `RandomHorizontalFlip`.
- Two-head model: shared backbone with separate fine (100-way) and coarse (20-way) classifiers.

## Baselines
- **uncertainty** — Kendall et al., arXiv:1705.07115; learns one log-variance per task initialized to `0`.
- **dwa** — Liu et al., arXiv:1803.10704; default temperature `T=2.0`.
- **pcgrad** — Yu et al., arXiv:2001.06782; project conflicting fine/coarse gradients on each parameter group.

## Implementation Constraint
The combination module must remain differentiable and must not change labels, heads, datasets, model backbones, or the outer training loop.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/pytorch-vision/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `pytorch-vision/custom_mtl.py`
- editable lines **195–216**




## Readable Context


### `pytorch-vision/custom_mtl.py`  [EDITABLE — lines 195–216 only]

```python
     1: """CV Multi-Task Loss Benchmark.
     2: 
     3: Train vision models (ResNet, VGG) on CIFAR-100 with TWO classification heads
     4: (fine: 100 classes, coarse: 20 superclasses) to evaluate multi-task loss
     5: combination strategies.
     6: 
     7: FIXED: Model architectures, data pipeline, training loop.
     8: EDITABLE: MultiTaskLoss class.
     9: 
    10: Usage:
    11:     python custom_mtl.py --arch resnet20 --seed 42
    12: """
    13: 
    14: import argparse
    15: import math
    16: import os
    17: import time
    18: 
    19: import torch
    20: import torch.nn as nn
    21: import torch.nn.functional as F
    22: import torch.optim as optim
    23: import torchvision
    24: import torchvision.transforms as transforms
    25: 
    26: 
    27: # ============================================================================
    28: # CIFAR-100 Coarse Label Mapping (FIXED)
    29: # ============================================================================
    30: 
    31: # Maps each of the 100 fine classes to one of 20 coarse superclasses.
    32: # Source: CIFAR-100 dataset specification (Krizhevsky, 2009).
    33: CIFAR100_COARSE_MAP = [
    34:     4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    35:     3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    36:     6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    37:     0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    38:     5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    39:     16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    40:     10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    41:     2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    42:     16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    43:     18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
    44: ]
    45: 
    46: 
    47: # ============================================================================
    48: # Model Architectures with Two Heads (FIXED)
    49: # ============================================================================
    50: 
    51: class BasicBlock(nn.Module):
    52:     """Basic residual block for CIFAR ResNets."""
    53:     expansion = 1
    54: 
    55:     def __init__(self, in_planes, planes, stride=1):
    56:         super().__init__()
    57:         self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
    58:         self.bn1 = nn.BatchNorm2d(planes)
    59:         self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
    60:         self.bn2 = nn.BatchNorm2d(planes)
    61:         self.shortcut = nn.Sequential()
    62:         if stride != 1 or in_planes != planes * self.expansion:
    63:             self.shortcut = nn.Sequential(
    64:                 nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
    65:                 nn.BatchNorm2d(planes * self.expansion),
    66:             )
    67: 
    68:     def forward(self, x):
    69:         out = F.relu(self.bn1(self.conv1(x)))
    70:         out = self.bn2(self.conv2(out))
    71:         out += self.shortcut(x)
    72:         return F.relu(out)
    73: 
    74: 
    75: class ResNet(nn.Module):
    76:     """CIFAR-adapted ResNet with two classification heads.
    77: 
    78:     Uses 3x3 initial conv (no 7x7), no max pooling, global avg pool at end.
    79:     Standard depths: ResNet-20 ([3,3,3]), ResNet-56 ([9,9,9]).
    80:     Two heads: fc_fine (100 classes) and fc_coarse (20 superclasses).
    81:     """
    82: 
    83:     def __init__(self, block, num_blocks):
    84:         super().__init__()
    85:         self.in_planes = 16
    86:         self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
    87:         self.bn1 = nn.BatchNorm2d(16)
    88:         self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
    89:         self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
    90:         self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
    91:         self.fc_fine = nn.Linear(64 * block.expansion, 100)
    92:         self.fc_coarse = nn.Linear(64 * block.expansion, 20)
    93: 
    94:     def _make_layer(self, block, planes, num_blocks, stride):
    95:         strides = [stride] + [1] * (num_blocks - 1)
    96:         layers = []
    97:         for s in strides:
    98:             layers.append(block(self.in_planes, planes, s))
    99:             self.in_planes = planes * block.expansion
   100:         return nn.Sequential(*layers)
   101: 
   102:     def forward(self, x):
   103:         out = F.relu(self.bn1(self.conv1(x)))
   104:         out = self.layer1(out)
   105:         out = self.layer2(out)
   106:         out = self.layer3(out)
   107:         out = F.adaptive_avg_pool2d(out, 1)
   108:         out = out.view(out.size(0), -1)
   109:         return self.fc_fine(out), self.fc_coarse(out)
   110: 
   111: 
   112: class VGG(nn.Module):
   113:     """VGG-16 with BatchNorm, adapted for CIFAR, with two classification heads.
   114: 
   115:     Uses adaptive avg pool instead of large FC layers, suitable for 32x32 input.
   116:     Two heads: fine (100 classes) and coarse (20 superclasses).
   117:     """
   118: 
   119:     VGG16_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
   120:                  512, 512, 512, 'M', 512, 512, 512, 'M']
   121: 
   122:     def __init__(self):
   123:         super().__init__()
   124:         self.features = self._make_layers(self.VGG16_CFG)
   125:         self.classifier_fine = nn.Sequential(
   126:             nn.Linear(512, 512),
   127:             nn.ReLU(True),
   128:             nn.Dropout(0.5),
   129:             nn.Linear(512, 100),
   130:         )
   131:         self.classifier_coarse = nn.Sequential(
   132:             nn.Linear(512, 512),
   133:             nn.ReLU(True),
   134:             nn.Dropout(0.5),
   135:             nn.Linear(512, 20),
   136:         )
   137: 
   138:     def _make_layers(self, cfg):
   139:         layers = []
   140:         in_channels = 3
   141:         for v in cfg:
   142:             if v == 'M':
   143:                 layers.append(nn.MaxPool2d(2, 2))
   144:             else:
   145:                 layers += [
   146:                     nn.Conv2d(in_channels, v, 3, padding=1),
   147:                     nn.BatchNorm2d(v),
   148:                     nn.ReLU(inplace=True),
   149:                 ]
   150:                 in_channels = v
   151:         return nn.Sequential(*layers)
   152: 
   153:     def forward(self, x):
   154:         x = self.features(x)
   155:         x = F.adaptive_avg_pool2d(x, 1)
   156:         x = x.view(x.size(0), -1)
   157:         return self.classifier_fine(x), self.classifier_coarse(x)
   158: 
   159: 
   160: def build_model(arch):
   161:     """Build model by architecture name (always CIFAR-100 two-head)."""
   162:     if arch == 'resnet20':
   163:         return ResNet(BasicBlock, [3, 3, 3])
   164:     elif arch == 'resnet56':
   165:         return ResNet(BasicBlock, [9, 9, 9])
   166:     elif arch == 'vgg16bn':
   167:         return VGG()
   168:     else:
   169:         raise ValueError(f"Unknown architecture: {arch}")
   170: 
   171: 
   172: # ============================================================================
   173: # Weight Initialization (FIXED)
   174: # ============================================================================
   175: 
   176: def initialize_weights(model):
   177:     """Kaiming initialization for all layers."""
   178:     for m in model.modules():
   179:         if isinstance(m, nn.Conv2d):
   180:             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
   181:         elif isinstance(m, nn.BatchNorm2d):
   182:             nn.init.constant_(m.weight, 1)
   183:             nn.init.constant_(m.bias, 0)
   184:         elif isinstance(m, nn.Linear):
   185:             nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
   186:             if m.bias is not None:
   187:                 nn.init.constant_(m.bias, 0)
   188: 
   189: 
   190: # ============================================================================
   191: # EDITABLE
   192: # ============================================================================
   193: 
   194: # -- EDITABLE REGION START (lines 195-216) ------------------------------------
   195: class MultiTaskLoss(nn.Module):
   196:     """Multi-task loss combination for fine + coarse classification.
   197: 
   198:     Args:
   199:         num_tasks: int (always 2)
   200:     """
   201: 
   202:     def __init__(self, num_tasks=2):
   203:         super().__init__()
   204: 
   205:     def forward(self, fine_loss, coarse_loss, epoch, total_epochs):
   206:         """Combine fine and coarse classification losses.
   207: 
   208:         Args:
   209:             fine_loss: scalar tensor, CE loss for 100-class fine prediction
   210:             coarse_loss: scalar tensor, CE loss for 20-class coarse prediction
   211:             epoch: int, current epoch (0-indexed)
   212:             total_epochs: int, total number of training epochs
   213:         Returns:
   214:             combined scalar loss
   215:         """
   216:         return fine_loss + coarse_loss
   217: # -- EDITABLE REGION END (lines 195-216) --------------------------------------
   218: 
   219: 
   220: # ============================================================================
   221: # Data Loading (FIXED)
   222: # ============================================================================
   223: 
   224: class CIFAR100MultiTask(torch.utils.data.Dataset):
   225:     """Wraps CIFAR-100 to return (image, fine_label, coarse_label) tuples."""
   226: 
   227:     def __init__(self, root, train=True, transform=None, download=False):
   228:         self.dataset = torchvision.datasets.CIFAR100(
   229:             root=root, train=train, transform=transform, download=download,
   230:         )
   231:         self.coarse_map = torch.tensor(CIFAR100_COARSE_MAP, dtype=torch.long)
   232: 
   233:     def __len__(self):
   234:         return len(self.dataset)
   235: 
   236:     def __getitem__(self, idx):
   237:         image, fine_label = self.dataset[idx]
   238:         coarse_label = self.coarse_map[fine_label].item()
   239:         return image, fine_label, coarse_label
   240: 
   241: 
   242: def get_dataloaders(data_root, batch_size=128, num_workers=4):
   243:     """Create CIFAR-100 multi-task train/test dataloaders."""
   244:     mean = (0.5071, 0.4867, 0.4408)
   245:     std = (0.2675, 0.2565, 0.2761)
   246: 
   247:     train_transform = transforms.Compose([
   248:         transforms.RandomCrop(32, padding=4),
   249:         transforms.RandomHorizontalFlip(),
   250:         transforms.ToTensor(),
   251:         transforms.Normalize(mean, std),
   252:     ])
   253:     test_transform = transforms.Compose([
   254:         transforms.ToTensor(),
   255:         transforms.Normalize(mean, std),
   256:     ])
   257: 
   258:     train_set = CIFAR100MultiTask(
   259:         root=data_root, train=True, transform=train_transform, download=False,
   260:     )
   261:     test_set = CIFAR100MultiTask(
   262:         root=data_root, train=False, transform=test_transform, download=False,
   263:     )
   264: 
   265:     train_loader = torch.utils.data.DataLoader(
   266:         train_set, batch_size=batch_size, shuffle=True,
   267:         num_workers=num_workers, pin_memory=True,
   268:     )
   269:     test_loader = torch.utils.data.DataLoader(
   270:         test_set, batch_size=batch_size, shuffle=False,
   271:         num_workers=num_workers, pin_memory=True,
   272:     )
   273:     return train_loader, test_loader
   274: 
   275: 
   276: # ============================================================================
   277: # Training Loop (FIXED)
   278: # ============================================================================
   279: 
   280: def train_epoch(model, loader, mtl_loss, optimizer, device, epoch, total_epochs):
   281:     """Train for one epoch with multi-task loss. Returns (avg_loss, fine_accuracy%)."""
   282:     model.train()
   283:     mtl_loss.train()
   284:     total_loss, correct, total = 0.0, 0, 0
   285:     for inputs, fine_targets, coarse_targets in loader:
   286:         inputs = inputs.to(device)
   287:         fine_targets = fine_targets.to(device)
   288:         coarse_targets = coarse_targets.to(device)
   289: 
   290:         optimizer.zero_grad()
   291:         fine_logits, coarse_logits = model(inputs)
   292:         fine_loss = F.cross_entropy(fine_logits, fine_targets)
   293:         coarse_loss = F.cross_entropy(coarse_logits, coarse_targets)
   294:         loss = mtl_loss(fine_loss, coarse_loss, epoch, total_epochs)
   295:         loss.backward()
   296:         optimizer.step()
   297: 
   298:         total_loss += loss.item() * inputs.size(0)
   299:         _, predicted = fine_logits.max(1)
   300:         correct += predicted.eq(fine_targets).sum().item()
   301:         total += inputs.size(0)
   302:     return total_loss / total, 100.0 * correct / total
   303: 
   304: 
   305: def evaluate(model, loader, device):
   306:     """Evaluate on test set. Returns fine-class accuracy%."""
   307:     model.eval()
   308:     correct, total = 0, 0
   309:     with torch.no_grad():
   310:         for inputs, fine_targets, coarse_targets in loader:
   311:             inputs = inputs.to(device)
   312:             fine_targets = fine_targets.to(device)
   313:             fine_logits, _ = model(inputs)
   314:             _, predicted = fine_logits.max(1)
   315:             correct += predicted.eq(fine_targets).sum().item()
   316:             total += inputs.size(0)
   317:     return 100.0 * correct / total
   318: 
   319: 
   320: # ============================================================================
   321: # Main (FIXED)
   322: # ============================================================================
   323: 
   324: def main():
   325:     parser = argparse.ArgumentParser(description="CV Multi-Task Loss Benchmark")
   326:     parser.add_argument('--arch', type=str, required=True,
   327:                         choices=['resnet20', 'resnet56', 'vgg16bn'])
   328:     parser.add_argument('--data-root', type=str, default='/data/cifar')
   329:     parser.add_argument('--epochs', type=int, default=200)
   330:     parser.add_argument('--batch-size', type=int, default=128)
   331:     parser.add_argument('--lr', type=float, default=0.1)
   332:     parser.add_argument('--momentum', type=float, default=0.9)
   333:     parser.add_argument('--weight-decay', type=float, default=5e-4)
   334:     parser.add_argument('--seed', type=int, default=42)
   335:     parser.add_argument('--output-dir', type=str, default='.')
   336:     args = parser.parse_args()
   337: 
   338:     torch.manual_seed(args.seed)
   339:     torch.cuda.manual_seed_all(args.seed)
   340:     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   341: 
   342:     # Data
   343:     train_loader, test_loader = get_dataloaders(
   344:         args.data_root, args.batch_size,
   345:     )
   346: 
   347:     # Model
   348:     model = build_model(args.arch)
   349:     initialize_weights(model)
   350:     model = model.to(device)
   351: 
   352:     # Multi-task loss
   353:     mtl_loss = MultiTaskLoss(num_tasks=2).to(device)
   354: 
   355:     # Optimizer — include mtl_loss parameters (e.g. learnable weights)
   356:     all_params = list(model.parameters()) + list(mtl_loss.parameters())
   357:     optimizer = optim.SGD(
   358:         all_params, lr=args.lr,
   359:         momentum=args.momentum, weight_decay=args.weight_decay,
   360:     )
   361:     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
   362: 
   363:     # Train
   364:     best_acc = 0.0
   365:     for epoch in range(args.epochs):
   366:         train_loss, train_acc = train_epoch(
   367:             model, train_loader, mtl_loss, optimizer, device, epoch, args.epochs,
   368:         )
   369:         test_acc = evaluate(model, test_loader, device)
   370:         scheduler.step()
   371: 
   372:         if (epoch + 1) % 10 == 0 or epoch == 0:
   373:             print(
   374:                 f"TRAIN_METRICS: epoch={epoch+1} train_loss={train_loss:.4f} "
   375:                 f"train_acc={train_acc:.2f} test_acc={test_acc:.2f} "
   376:                 f"lr={optimizer.param_groups[0]['lr']:.6f}",
   377:                 flush=True,
   378:             )
   379: 
   380:         if test_acc > best_acc:
   381:             best_acc = test_acc
   382: 
   383:     print(f"TEST_METRICS: test_acc={best_acc:.2f}", flush=True)
   384: 
   385: 
   386: if __name__ == '__main__':
   387:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `uncertainty` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_mtl.py`:

```python
Lines 195–211:
   192: # ============================================================================
   193: 
   194: # -- EDITABLE REGION START (lines 195-216) ------------------------------------
   195: class MultiTaskLoss(nn.Module):
   196:     """Uncertainty weighting (Kendall et al., 2018).
   197: 
   198:     Learns per-task log-variance: loss_i / exp(log_var_i) + log_var_i.
   199:     """
   200: 
   201:     def __init__(self, num_tasks=2):
   202:         super().__init__()
   203:         self.log_vars = nn.Parameter(torch.zeros(num_tasks))
   204: 
   205:     def forward(self, fine_loss, coarse_loss, epoch, total_epochs):
   206:         losses = [fine_loss, coarse_loss]
   207:         total = sum(
   208:             torch.exp(-self.log_vars[i]) * losses[i] + self.log_vars[i]
   209:             for i in range(2)
   210:         )
   211:         return total
   212: # -- EDITABLE REGION END (lines 195-216) --------------------------------------
   213: 
   214: 
```

### `dwa` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_mtl.py`:

```python
Lines 195–214:
   192: # ============================================================================
   193: 
   194: # -- EDITABLE REGION START (lines 195-216) ------------------------------------
   195: class MultiTaskLoss(nn.Module):
   196:     """Dynamic Weight Average (Liu et al., 2019).
   197: 
   198:     Weights tasks by relative loss change rate with temperature.
   199:     """
   200: 
   201:     def __init__(self, num_tasks=2):
   202:         super().__init__()
   203:         self.prev_losses = None
   204:         self.T = 2.0  # temperature
   205: 
   206:     def forward(self, fine_loss, coarse_loss, epoch, total_epochs):
   207:         losses = torch.stack([fine_loss, coarse_loss])
   208:         if self.prev_losses is None or epoch == 0:
   209:             weights = torch.ones(2, device=losses.device)
   210:         else:
   211:             ratios = losses.detach() / (self.prev_losses + 1e-8)
   212:             weights = 2 * F.softmax(ratios / self.T, dim=0)
   213:         self.prev_losses = losses.detach().clone()
   214:         return (weights * losses).sum()
   215: # -- EDITABLE REGION END (lines 195-216) --------------------------------------
   216: 
   217: 
```

### `pcgrad` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_mtl.py`:

```python
Lines 195–280:
   192: # ============================================================================
   193: 
   194: # -- EDITABLE REGION START (lines 195-216) ------------------------------------
   195: class MultiTaskLoss(nn.Module):
   196:     """PCGrad: Gradient Surgery for Multi-Task Learning (Yu et al., 2020).
   197: 
   198:     Projects conflicting task gradients onto the normal plane of the
   199:     other when their cosine similarity is negative, reducing gradient
   200:     interference between tasks.
   201:     """
   202: 
   203:     def __init__(self, num_tasks=2):
   204:         super().__init__()
   205:         self._shared_params = None
   206: 
   207:     def _get_shared_params(self, loss):
   208:         """Extract shared model parameters from the computation graph."""
   209:         if self._shared_params is not None:
   210:             return self._shared_params
   211:         # Walk the computation graph to find leaf parameters
   212:         params = []
   213:         seen = set()
   214:         def _walk(grad_fn):
   215:             if grad_fn is None:
   216:                 return
   217:             for child, _ in grad_fn.next_functions:
   218:                 if child is None:
   219:                     continue
   220:                 cid = id(child)
   221:                 if cid in seen:
   222:                     continue
   223:                 seen.add(cid)
   224:                 if hasattr(child, 'variable'):
   225:                     p = child.variable
   226:                     if p.requires_grad:
   227:                         params.append(p)
   228:                 _walk(child)
   229:         _walk(loss.grad_fn)
   230:         self._shared_params = params
   231:         return params
   232: 
   233:     def forward(self, fine_loss, coarse_loss, epoch, total_epochs):
   234:         params = self._get_shared_params(fine_loss)
   235:         if len(params) == 0:
   236:             return fine_loss + coarse_loss
   237: 
   238:         # Compute per-task gradients
   239:         grads_fine = torch.autograd.grad(
   240:             fine_loss, params, retain_graph=True, allow_unused=True,
   241:         )
   242:         grads_coarse = torch.autograd.grad(
   243:             coarse_loss, params, retain_graph=True, allow_unused=True,
   244:         )
   245: 
   246:         # Flatten gradients into vectors
   247:         g0 = torch.cat([
   248:             g.flatten() if g is not None else torch.zeros_like(p).flatten()
   249:             for g, p in zip(grads_fine, params)
   250:         ])
   251:         g1 = torch.cat([
   252:             g.flatten() if g is not None else torch.zeros_like(p).flatten()
   253:             for g, p in zip(grads_coarse, params)
   254:         ])
   255: 
   256:         # PCGrad: project conflicting gradients when cosine similarity < 0
   257:         dot = torch.dot(g0, g1)
   258:         if dot < 0:
   259:             # Project each gradient onto the normal plane of the other.
   260:             # Use originals for both projections (symmetric dot product).
   261:             g0_norm_sq = torch.dot(g0, g0) + 1e-12
   262:             g1_norm_sq = torch.dot(g1, g1) + 1e-12
   263:             g0_proj = g0 - (dot / g1_norm_sq) * g1
   264:             g1_proj = g1 - (dot / g0_norm_sq) * g0
   265:             g0 = g0_proj
   266:             g1 = g1_proj
   267: 
   268:         # Combined projected gradient
   269:         g_pcgrad = g0 + g1
   270: 
   271:         # Construct a surrogate loss whose gradient equals g_pcgrad.
   272:         # loss = sum_i (g_pcgrad_i * param_i), so grad w.r.t. param_i = g_pcgrad_i
   273:         offset = 0
   274:         surrogate = torch.tensor(0.0, device=fine_loss.device)
   275:         for p in params:
   276:             numel = p.numel()
   277:             chunk = g_pcgrad[offset:offset + numel].reshape(p.shape).detach()
   278:             surrogate = surrogate + (chunk * p).sum()
   279:             offset += numel
   280:         return surrogate
   281: # -- EDITABLE REGION END (lines 195-216) --------------------------------------
   282: 
   283: 
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
