# MLS-Bench: dl-residual-connection

# DL Residual Connection Block Design

## Research Question
Design a residual / skip-connection block for CIFAR-style ResNets that improves classification performance across different network depths and datasets, while keeping the broader training recipe, initialization, data pipeline, optimizer, and classifier objective fixed.

## Background
Residual connections (He et al., "Deep Residual Learning for Image Recognition", arXiv:1512.03385) enabled training of very deep networks by providing identity shortcut paths. The basic residual block adds the input to the output of two stacked 3×3 convolutions. Several improvements have been proposed:

- **Pre-activation ResBlock** (He et al., "Identity Mappings in Deep Residual Networks", ECCV 2016, arXiv:1603.05027): BN-ReLU-Conv ordering, enabling cleaner gradient flow through identity shortcuts.
- **ReZero / gated residual** (Bachlechner et al., "ReZero is All You Need: Fast Convergence at Large Depth", arXiv:2003.04887): a single learnable scalar gate, initialized to `0`, multiplies the residual branch before addition; the network gradually learns the optimal residual contribution per block.
- **Stochastic Depth** (Huang et al., ECCV 2016, arXiv:1603.09382): randomly drops entire residual blocks during training with a linearly decaying survival probability `p_l = 1 − (l/L)(1 − p_L)` and final-block survival `p_L=0.5`; acts as an implicit ensemble regularizer especially effective for very deep networks.
- **ResNeXt** (Xie et al., CVPR 2017, arXiv:1611.05431): grouped convolutions for multi-branch aggregation, with cardinality as a third capacity axis.
- **Res2Net** (Gao et al., TPAMI 2019, arXiv:1904.01169): hierarchical residual-like connections within a single block for multi-scale feature extraction.
- **SE block** (Hu, Shen & Sun, CVPR 2018): channel attention applied inside the residual branch.

There is room for novel block designs that better balance gradient flow, feature reuse, and regularization, particularly across varying network depths.

## What You Can Modify
The `CustomBlock` class inside `pytorch-vision/custom_residual.py`. It is the residual block used by the ResNet backbone.

Constraints (the backbone relies on these):
- Constructor: `CustomBlock(in_planes, planes, stride)`.
- Class attribute `expansion` (`1` for basic, `4` for bottleneck, etc.).
- `forward(x)` returns a tensor with `planes * expansion` channels.
- The shortcut must handle dimension mismatches when `stride != 1` or when the input/output channel count differs.

You may modify the internal convolution structure (number, kernel sizes, grouping), activation/normalization placement and type, the shortcut/skip design, attention mechanisms (channel or spatial), the `expansion` attribute, and any additional modules within the block.

## Fixed Pipeline
- Optimizer: SGD with `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`.
- Schedule: cosine annealing over `200` epochs.
- Data augmentation: `RandomCrop(32, pad=4)` + `RandomHorizontalFlip`.
- Network architecture: CIFAR-adapted ResNets at varying depths, evaluated on CIFAR-style image classification datasets.

## Baselines
- **pre_activation** — He et al., arXiv:1603.05027; BN-ReLU-Conv ordering inside the block.
- **gated_residual** — ReZero-style learnable scalar gate per block, initialized to `0` (Bachlechner et al., arXiv:2003.04887).
- **stochastic_depth** — Huang et al., arXiv:1603.09382; linearly decaying per-block survival probability with `p_L=0.5`.

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/pytorch-vision/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `pytorch-vision/custom_residual.py`
- editable lines **31–61**




## Readable Context


### `pytorch-vision/custom_residual.py`  [EDITABLE — lines 31–61 only]

```python
     1: """CV Residual Connection Benchmark.
     2: 
     3: Train CIFAR ResNets with custom residual blocks to evaluate
     4: skip/residual connection designs.
     5: 
     6: FIXED: ResNet backbone, data pipeline, training loop.
     7: EDITABLE: CustomBlock class (residual block design).
     8: 
     9: Usage:
    10:     python custom_residual.py --arch resnet20 --dataset cifar10 --seed 42
    11: """
    12: 
    13: import argparse
    14: import math
    15: import os
    16: import time
    17: 
    18: import torch
    19: import torch.nn as nn
    20: import torch.nn.functional as F
    21: import torch.optim as optim
    22: import torchvision
    23: import torchvision.transforms as transforms
    24: 
    25: 
    26: # ============================================================================
    27: # Residual Block
    28: # ============================================================================
    29: 
    30: # -- EDITABLE REGION START (lines 31-61) ------------------------------------
    31: class CustomBlock(nn.Module):
    32:     """Custom residual block for CIFAR ResNets.
    33: 
    34:     Args:
    35:         in_planes: input channels
    36:         planes: output channels
    37:         stride: spatial stride (1 or 2)
    38: 
    39:     Must set class attribute `expansion = 1` (or 4 for bottleneck).
    40:     The shortcut dimension must match planes * expansion.
    41:     """
    42:     expansion = 1
    43: 
    44:     def __init__(self, in_planes, planes, stride=1):
    45:         super().__init__()
    46:         self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
    47:         self.bn1 = nn.BatchNorm2d(planes)
    48:         self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
    49:         self.bn2 = nn.BatchNorm2d(planes)
    50:         self.shortcut = nn.Sequential()
    51:         if stride != 1 or in_planes != planes * self.expansion:
    52:             self.shortcut = nn.Sequential(
    53:                 nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
    54:                 nn.BatchNorm2d(planes * self.expansion),
    55:             )
    56: 
    57:     def forward(self, x):
    58:         out = F.relu(self.bn1(self.conv1(x)))
    59:         out = self.bn2(self.conv2(out))
    60:         out += self.shortcut(x)
    61:         return F.relu(out)
    62: # -- EDITABLE REGION END (lines 31-61) --------------------------------------
    63: 
    64: 
    65: # ============================================================================
    66: # ResNet Architecture (FIXED)
    67: # ============================================================================
    68: 
    69: class ResNet(nn.Module):
    70:     """CIFAR-adapted ResNet using CustomBlock.
    71: 
    72:     Uses 3x3 initial conv (no 7x7), no max pooling, global avg pool at end.
    73:     Standard depths: ResNet-20 ([3,3,3]), ResNet-56 ([9,9,9]), ResNet-110 ([18,18,18]).
    74:     """
    75: 
    76:     def __init__(self, block, num_blocks, num_classes=10):
    77:         super().__init__()
    78:         self.in_planes = 16
    79:         self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
    80:         self.bn1 = nn.BatchNorm2d(16)
    81:         self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
    82:         self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
    83:         self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
    84:         self.fc = nn.Linear(64 * block.expansion, num_classes)
    85: 
    86:     def _make_layer(self, block, planes, num_blocks, stride):
    87:         strides = [stride] + [1] * (num_blocks - 1)
    88:         layers = []
    89:         for s in strides:
    90:             layers.append(block(self.in_planes, planes, s))
    91:             self.in_planes = planes * block.expansion
    92:         return nn.Sequential(*layers)
    93: 
    94:     def forward(self, x):
    95:         out = F.relu(self.bn1(self.conv1(x)))
    96:         out = self.layer1(out)
    97:         out = self.layer2(out)
    98:         out = self.layer3(out)
    99:         out = F.adaptive_avg_pool2d(out, 1)
   100:         out = out.view(out.size(0), -1)
   101:         return self.fc(out)
   102: 
   103: 
   104: def build_model(arch, num_classes):
   105:     """Build model by architecture name."""
   106:     if arch == 'resnet20':
   107:         return ResNet(CustomBlock, [3, 3, 3], num_classes)
   108:     elif arch == 'resnet56':
   109:         return ResNet(CustomBlock, [9, 9, 9], num_classes)
   110:     elif arch == 'resnet110':
   111:         return ResNet(CustomBlock, [18, 18, 18], num_classes)
   112:     else:
   113:         raise ValueError(f"Unknown architecture: {arch}")
   114: 
   115: 
   116: # ============================================================================
   117: # Weight Initialization (FIXED)
   118: # ============================================================================
   119: 
   120: def initialize_weights(model):
   121:     """Standard Kaiming initialization for all layers."""
   122:     for m in model.modules():
   123:         if isinstance(m, nn.Conv2d):
   124:             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
   125:         elif isinstance(m, nn.BatchNorm2d):
   126:             nn.init.constant_(m.weight, 1)
   127:             nn.init.constant_(m.bias, 0)
   128:         elif isinstance(m, nn.Linear):
   129:             nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
   130:             if m.bias is not None:
   131:                 nn.init.constant_(m.bias, 0)
   132: 
   133: 
   134: # ============================================================================
   135: # Data Loading (FIXED)
   136: # ============================================================================
   137: 
   138: def get_dataloaders(dataset, data_root, batch_size=128, num_workers=4):
   139:     """Create CIFAR train/test dataloaders with standard augmentation."""
   140:     if dataset == 'cifar10':
   141:         mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
   142:         num_classes = 10
   143:         Dataset = torchvision.datasets.CIFAR10
   144:     elif dataset == 'cifar100':
   145:         mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
   146:         num_classes = 100
   147:         Dataset = torchvision.datasets.CIFAR100
   148:     else:
   149:         raise ValueError(f"Unknown dataset: {dataset}")
   150: 
   151:     train_transform = transforms.Compose([
   152:         transforms.RandomCrop(32, padding=4),
   153:         transforms.RandomHorizontalFlip(),
   154:         transforms.ToTensor(),
   155:         transforms.Normalize(mean, std),
   156:     ])
   157:     test_transform = transforms.Compose([
   158:         transforms.ToTensor(),
   159:         transforms.Normalize(mean, std),
   160:     ])
   161: 
   162:     train_set = Dataset(root=data_root, train=True, download=False, transform=train_transform)
   163:     test_set = Dataset(root=data_root, train=False, download=False, transform=test_transform)
   164: 
   165:     train_loader = torch.utils.data.DataLoader(
   166:         train_set, batch_size=batch_size, shuffle=True,
   167:         num_workers=num_workers, pin_memory=True,
   168:     )
   169:     test_loader = torch.utils.data.DataLoader(
   170:         test_set, batch_size=batch_size, shuffle=False,
   171:         num_workers=num_workers, pin_memory=True,
   172:     )
   173:     return train_loader, test_loader, num_classes
   174: 
   175: 
   176: # ============================================================================
   177: # Training Loop (FIXED)
   178: # ============================================================================
   179: 
   180: def train_epoch(model, loader, criterion, optimizer, device):
   181:     """Train for one epoch. Returns (avg_loss, accuracy%)."""
   182:     model.train()
   183:     total_loss, correct, total = 0.0, 0, 0
   184:     for inputs, targets in loader:
   185:         inputs, targets = inputs.to(device), targets.to(device)
   186:         optimizer.zero_grad()
   187:         outputs = model(inputs)
   188:         loss = criterion(outputs, targets)
   189:         loss.backward()
   190:         optimizer.step()
   191:         total_loss += loss.item() * inputs.size(0)
   192:         _, predicted = outputs.max(1)
   193:         correct += predicted.eq(targets).sum().item()
   194:         total += inputs.size(0)
   195:     return total_loss / total, 100.0 * correct / total
   196: 
   197: 
   198: def evaluate(model, loader, criterion, device):
   199:     """Evaluate on test set. Returns (avg_loss, accuracy%)."""
   200:     model.eval()
   201:     total_loss, correct, total = 0.0, 0, 0
   202:     with torch.no_grad():
   203:         for inputs, targets in loader:
   204:             inputs, targets = inputs.to(device), targets.to(device)
   205:             outputs = model(inputs)
   206:             loss = criterion(outputs, targets)
   207:             total_loss += loss.item() * inputs.size(0)
   208:             _, predicted = outputs.max(1)
   209:             correct += predicted.eq(targets).sum().item()
   210:             total += inputs.size(0)
   211:     return total_loss / total, 100.0 * correct / total
   212: 
   213: 
   214: def main():
   215:     parser = argparse.ArgumentParser(description="CV Residual Connection Benchmark")
   216:     parser.add_argument('--arch', type=str, required=True,
   217:                         choices=['resnet20', 'resnet56', 'resnet110'])
   218:     parser.add_argument('--dataset', type=str, required=True,
   219:                         choices=['cifar10', 'cifar100'])
   220:     parser.add_argument('--data-root', type=str, default='/data/cifar')
   221:     parser.add_argument('--epochs', type=int, default=200)
   222:     parser.add_argument('--batch-size', type=int, default=128)
   223:     parser.add_argument('--lr', type=float, default=0.1)
   224:     parser.add_argument('--momentum', type=float, default=0.9)
   225:     parser.add_argument('--weight-decay', type=float, default=5e-4)
   226:     parser.add_argument('--seed', type=int, default=42)
   227:     parser.add_argument('--output-dir', type=str, default='.')
   228:     args = parser.parse_args()
   229: 
   230:     torch.manual_seed(args.seed)
   231:     torch.cuda.manual_seed_all(args.seed)
   232:     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   233: 
   234:     # Data
   235:     train_loader, test_loader, num_classes = get_dataloaders(
   236:         args.dataset, args.data_root, args.batch_size,
   237:     )
   238: 
   239:     # Model
   240:     model = build_model(args.arch, num_classes)
   241: 
   242:     # Initialize
   243:     initialize_weights(model)
   244:     model = model.to(device)
   245: 
   246:     # Optimizer
   247:     criterion = nn.CrossEntropyLoss()
   248:     optimizer = optim.SGD(
   249:         model.parameters(), lr=args.lr,
   250:         momentum=args.momentum, weight_decay=args.weight_decay,
   251:     )
   252:     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
   253: 
   254:     # Train
   255:     best_acc = 0.0
   256:     for epoch in range(args.epochs):
   257:         train_loss, train_acc = train_epoch(
   258:             model, train_loader, criterion, optimizer, device,
   259:         )
   260:         test_loss, test_acc = evaluate(model, test_loader, criterion, device)
   261:         scheduler.step()
   262: 
   263:         if (epoch + 1) % 10 == 0 or epoch == 0:
   264:             print(
   265:                 f"TRAIN_METRICS: epoch={epoch+1} train_loss={train_loss:.4f} "
   266:                 f"train_acc={train_acc:.2f} test_loss={test_loss:.4f} "
   267:                 f"test_acc={test_acc:.2f} lr={optimizer.param_groups[0]['lr']:.6f}",
   268:                 flush=True,
   269:             )
   270: 
   271:         if test_acc > best_acc:
   272:             best_acc = test_acc
   273: 
   274:     print(f"TEST_METRICS: test_acc={best_acc:.2f}", flush=True)
   275: 
   276: 
   277: if __name__ == '__main__':
   278:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `pre_activation` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_residual.py`:

```python
Lines 31–59:
    28: # ============================================================================
    29: 
    30: # -- EDITABLE REGION START (lines 31-61) ------------------------------------
    31: class CustomBlock(nn.Module):
    32:     """Pre-activation residual block (He et al., 2016 v2).
    33: 
    34:     Uses BN-ReLU-Conv order for cleaner gradient flow.
    35:     Both main branch and shortcut share the same pre-activation.
    36:     Residual scaling (alpha, init=0.1) stabilizes very deep networks.
    37:     """
    38:     expansion = 1
    39: 
    40:     def __init__(self, in_planes, planes, stride=1):
    41:         super().__init__()
    42:         self.bn1 = nn.BatchNorm2d(in_planes)
    43:         self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
    44:         self.bn2 = nn.BatchNorm2d(planes)
    45:         self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
    46:         self.alpha = nn.Parameter(torch.tensor(0.1))
    47:         self.downsample = None
    48:         if stride != 1 or in_planes != planes * self.expansion:
    49:             self.downsample = nn.Sequential(
    50:                 nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
    51:                 nn.BatchNorm2d(planes * self.expansion),
    52:             )
    53: 
    54:     def forward(self, x):
    55:         pre = F.relu(self.bn1(x))
    56:         out = self.conv1(pre)
    57:         out = self.conv2(F.relu(self.bn2(out)))
    58:         shortcut = self.downsample(x) if self.downsample is not None else x
    59:         return shortcut + self.alpha * out
    60: # -- EDITABLE REGION END (lines 31-61) --------------------------------------
    61: 
    62: 
```

### `gated_residual` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_residual.py`:

```python
Lines 31–59:
    28: # ============================================================================
    29: 
    30: # -- EDITABLE REGION START (lines 31-61) ------------------------------------
    31: class CustomBlock(nn.Module):
    32:     """Residual block with learnable residual gate (scalar scaling).
    33: 
    34:     A learnable parameter alpha scales the residual branch output before
    35:     adding to the shortcut: out = shortcut(x) + alpha * F(x).
    36:     Initialized at alpha=1.0 (standard residual behavior).
    37:     """
    38:     expansion = 1
    39: 
    40:     def __init__(self, in_planes, planes, stride=1):
    41:         super().__init__()
    42:         self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
    43:         self.bn1 = nn.BatchNorm2d(planes)
    44:         self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
    45:         self.bn2 = nn.BatchNorm2d(planes)
    46:         self.shortcut = nn.Sequential()
    47:         if stride != 1 or in_planes != planes * self.expansion:
    48:             self.shortcut = nn.Sequential(
    49:                 nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
    50:                 nn.BatchNorm2d(planes * self.expansion),
    51:             )
    52:         # Learnable residual gate initialized at 1.0
    53:         self.alpha = nn.Parameter(torch.ones(1))
    54: 
    55:     def forward(self, x):
    56:         out = F.relu(self.bn1(self.conv1(x)))
    57:         out = self.bn2(self.conv2(out))
    58:         out = self.shortcut(x) + self.alpha * out
    59:         return F.relu(out)
    60: # -- EDITABLE REGION END (lines 31-61) --------------------------------------
    61: 
    62: 
```

### `stochastic_depth` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_residual.py`:

```python
Lines 31–76:
    28: # ============================================================================
    29: 
    30: # -- EDITABLE REGION START (lines 31-61) ------------------------------------
    31: class CustomBlock(nn.Module):
    32:     """Residual block with stochastic depth (Huang et al., 2016).
    33: 
    34:     During training, each block's residual branch is randomly dropped with
    35:     probability (1 - survival_prob). The survival probability linearly decays
    36:     from 1.0 (first block) to p_L (last block). At test time, the residual
    37:     output is deterministically scaled by the survival probability.
    38:     """
    39:     expansion = 1
    40:     _block_counter = 0
    41:     _p_last = 0.5  # survival prob of the deepest block
    42: 
    43:     def __init__(self, in_planes, planes, stride=1):
    44:         super().__init__()
    45:         self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
    46:         self.bn1 = nn.BatchNorm2d(planes)
    47:         self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
    48:         self.bn2 = nn.BatchNorm2d(planes)
    49:         self.shortcut = nn.Sequential()
    50:         if stride != 1 or in_planes != planes * self.expansion:
    51:             self.shortcut = nn.Sequential(
    52:                 nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
    53:                 nn.BatchNorm2d(planes * self.expansion),
    54:             )
    55:         # Reset counter when first block of a new model is created
    56:         # (CIFAR ResNets always start layer1 with in_planes=16, planes=16, stride=1)
    57:         if in_planes == 16 and planes == 16 and stride == 1:
    58:             CustomBlock._block_counter = 0
    59:         CustomBlock._block_counter += 1
    60:         self.block_idx = CustomBlock._block_counter
    61: 
    62:     def forward(self, x):
    63:         shortcut = self.shortcut(x)
    64:         L = CustomBlock._block_counter
    65:         p = 1.0 - (self.block_idx / L) * (1.0 - CustomBlock._p_last)
    66:         if self.training:
    67:             if torch.rand(1).item() < p:
    68:                 out = F.relu(self.bn1(self.conv1(x)))
    69:                 out = self.bn2(self.conv2(out))
    70:                 return F.relu(out + shortcut)
    71:             else:
    72:                 return F.relu(shortcut)
    73:         else:
    74:             out = F.relu(self.bn1(self.conv1(x)))
    75:             out = self.bn2(self.conv2(out))
    76:             return F.relu(p * out + shortcut)
    77: # -- EDITABLE REGION END (lines 31-61) --------------------------------------
    78: 
    79: 
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
