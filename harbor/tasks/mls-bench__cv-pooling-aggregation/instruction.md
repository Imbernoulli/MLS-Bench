# MLS-Bench: cv-pooling-aggregation

# CV Global Pooling / Feature Aggregation Design

## Research Question
Design a global pooling / feature aggregation module for image classification that improves performance across different CNN architectures and datasets, while preserving the surrounding backbone and classifier interface.

## Background
Global pooling is the final spatial aggregation step in modern image-classification CNNs, reducing feature maps from `[B, C, H, W]` to `[B, C]` before the classifier head. The standard choice is Global Average Pooling (GAP), which computes the spatial mean per channel — simple and stable, but treats every spatial location identically and discards the distribution of activations. Alternatives include:

- **Global Max Pooling (GMP)**: selects the strongest activation per channel; captures peak features but ignores other spatial information.
- **Generalized Mean (GeM) Pooling** (Radenović, Tolias & Chum, "Fine-tuning CNN Image Retrieval with No Human Annotation", arXiv:1711.02512, TPAMI 2018): a learnable power-mean `f_p(x) = (mean(x^p))^(1/p)` that interpolates between average pooling (p=1) and max pooling (p→∞). The paper uses an initial value of `p=3.0`.
- **Average + Max**: element-wise sum (or concatenation reduced to `C`) of GAP and GMP, capturing both mean-field and peak statistics.
- Attention- or distribution-based aggregations that learn spatial weights or higher-order statistics.

There is room to design pooling rules that better capture spatial statistics of feature maps, adapt to different architectures, or learn task-specific aggregation patterns.

## What You Can Modify
The `CustomPool` class inside `pytorch-vision/custom_pool.py`. The forward signature takes a `[B, C, H, W]` tensor and must return a `[B, C]` tensor.

You may modify the aggregation function (mean, max, learned weights, attention, higher-order statistics), introduce learnable parameters, choose how spatial information is summarized (single-point, multi-scale, distribution-based), and apply channel-wise or spatial-wise weighting.

Constraints:
- Input shape: `[B, C, H, W]`. `C` varies by architecture (`64` for ResNet-56, `512` for VGG-16-BN, `1280` for MobileNetV2 at this resolution).
- Output shape: `[B, C]`. The output channel dimension must equal the input channel dimension exactly.
- Must work with variable spatial sizes (e.g. `8×8` for ResNet on CIFAR, `1×1` after VGG max-pools / MobileNetV2 stem).
- No access to training data or labels inside the pooling layer.

## Fixed Pipeline
The training and evaluation pipeline (data, augmentation, model, optimizer,
schedule, and metrics) is fixed by the harness and not editable.

## Baselines
- **global_max** — channel-wise max over the spatial axes (no extra parameters).
- **gem** — Radenović et al., arXiv:1711.02512; default learnable `p` initialized to `3.0`, with stability epsilon `1e-6`.
- **avg_max** — sum of GAP and GMP outputs (no learnable parameters).


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/pytorch-vision/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `pytorch-vision/custom_pool.py`
- editable lines **31–48**




## Readable Context


### `pytorch-vision/custom_pool.py`  [EDITABLE — lines 31–48 only]

```python
     1: """CV Pooling / Feature Aggregation Benchmark.
     2: 
     3: Train vision models (ResNet, VGG, MobileNetV2) on CIFAR-10/100/FashionMNIST to evaluate
     4: global pooling and feature aggregation strategies.
     5: 
     6: FIXED: Model architectures, data pipeline, training loop.
     7: EDITABLE: CustomPool class.
     8: 
     9: Usage:
    10:     python custom_pool.py --arch resnet20 --dataset cifar10 --seed 42
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
    27: # Global Pooling / Feature Aggregation
    28: # ============================================================================
    29: 
    30: # -- EDITABLE REGION START (lines 31-48) ------------------------------------
    31: class CustomPool(nn.Module):
    32:     """Custom global pooling layer.
    33: 
    34:     Reduces spatial feature maps [B, C, H, W] to feature vectors [B, C].
    35:     Used as the final spatial aggregation before the classifier head.
    36: 
    37:     Design considerations:
    38:         - How to aggregate spatial information (mean, max, learned, mixed)
    39:         - Whether to use learnable parameters for adaptive aggregation
    40:         - Robustness across different spatial resolutions and channel counts
    41:         - Interaction with downstream classifier and upstream features
    42:     """
    43: 
    44:     def __init__(self):
    45:         super().__init__()
    46: 
    47:     def forward(self, x):
    48:         return F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
    49: # -- EDITABLE REGION END (lines 31-48) --------------------------------------
    50: 
    51: 
    52: # ============================================================================
    53: # Model Architectures (FIXED)
    54: # ============================================================================
    55: 
    56: class BasicBlock(nn.Module):
    57:     """Basic residual block for CIFAR ResNets."""
    58:     expansion = 1
    59: 
    60:     def __init__(self, in_planes, planes, stride=1):
    61:         super().__init__()
    62:         self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
    63:         self.bn1 = nn.BatchNorm2d(planes)
    64:         self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
    65:         self.bn2 = nn.BatchNorm2d(planes)
    66:         self.shortcut = nn.Sequential()
    67:         if stride != 1 or in_planes != planes * self.expansion:
    68:             self.shortcut = nn.Sequential(
    69:                 nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
    70:                 nn.BatchNorm2d(planes * self.expansion),
    71:             )
    72: 
    73:     def forward(self, x):
    74:         out = F.relu(self.bn1(self.conv1(x)))
    75:         out = self.bn2(self.conv2(out))
    76:         out += self.shortcut(x)
    77:         return F.relu(out)
    78: 
    79: 
    80: class ResNet(nn.Module):
    81:     """CIFAR-adapted ResNet (He et al., 2016).
    82: 
    83:     Uses 3x3 initial conv (no 7x7), no max pooling, CustomPool at end.
    84:     Standard depths: ResNet-20 ([3,3,3]), ResNet-56 ([9,9,9]).
    85:     """
    86: 
    87:     def __init__(self, block, num_blocks, num_classes=10):
    88:         super().__init__()
    89:         self.in_planes = 16
    90:         self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
    91:         self.bn1 = nn.BatchNorm2d(16)
    92:         self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
    93:         self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
    94:         self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
    95:         self.pool = CustomPool()
    96:         self.fc = nn.Linear(64 * block.expansion, num_classes)
    97: 
    98:     def _make_layer(self, block, planes, num_blocks, stride):
    99:         strides = [stride] + [1] * (num_blocks - 1)
   100:         layers = []
   101:         for s in strides:
   102:             layers.append(block(self.in_planes, planes, s))
   103:             self.in_planes = planes * block.expansion
   104:         return nn.Sequential(*layers)
   105: 
   106:     def forward(self, x):
   107:         out = F.relu(self.bn1(self.conv1(x)))
   108:         out = self.layer1(out)
   109:         out = self.layer2(out)
   110:         out = self.layer3(out)
   111:         out = self.pool(out)
   112:         return self.fc(out)
   113: 
   114: 
   115: class VGG(nn.Module):
   116:     """VGG-16 with BatchNorm, adapted for CIFAR (Simonyan & Zisserman, 2015).
   117: 
   118:     Uses CustomPool instead of large FC layers, suitable for 32x32 input.
   119:     """
   120: 
   121:     VGG16_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
   122:                  512, 512, 512, 'M', 512, 512, 512, 'M']
   123: 
   124:     def __init__(self, num_classes=100):
   125:         super().__init__()
   126:         self.features = self._make_layers(self.VGG16_CFG)
   127:         self.pool = CustomPool()
   128:         self.classifier = nn.Sequential(
   129:             nn.Linear(512, 512),
   130:             nn.ReLU(True),
   131:             nn.Dropout(0.5),
   132:             nn.Linear(512, num_classes),
   133:         )
   134: 
   135:     def _make_layers(self, cfg):
   136:         layers = []
   137:         in_channels = 3
   138:         for v in cfg:
   139:             if v == 'M':
   140:                 layers.append(nn.MaxPool2d(2, 2))
   141:             else:
   142:                 layers += [
   143:                     nn.Conv2d(in_channels, v, 3, padding=1),
   144:                     nn.BatchNorm2d(v),
   145:                     nn.ReLU(inplace=True),
   146:                 ]
   147:                 in_channels = v
   148:         return nn.Sequential(*layers)
   149: 
   150:     def forward(self, x):
   151:         x = self.features(x)
   152:         x = self.pool(x)
   153:         return self.classifier(x)
   154: 
   155: 
   156: class InvertedResidual(nn.Module):
   157:     """MobileNetV2 inverted residual block (Sandler et al., 2018)."""
   158: 
   159:     def __init__(self, inp, oup, stride, expand_ratio):
   160:         super().__init__()
   161:         self.stride = stride
   162:         hidden = int(round(inp * expand_ratio))
   163:         self.use_res = (stride == 1 and inp == oup)
   164:         layers = []
   165:         if expand_ratio != 1:
   166:             layers += [
   167:                 nn.Conv2d(inp, hidden, 1, bias=False),
   168:                 nn.BatchNorm2d(hidden),
   169:                 nn.ReLU6(inplace=True),
   170:             ]
   171:         layers += [
   172:             nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
   173:             nn.BatchNorm2d(hidden),
   174:             nn.ReLU6(inplace=True),
   175:             nn.Conv2d(hidden, oup, 1, bias=False),
   176:             nn.BatchNorm2d(oup),
   177:         ]
   178:         self.conv = nn.Sequential(*layers)
   179: 
   180:     def forward(self, x):
   181:         if self.use_res:
   182:             return x + self.conv(x)
   183:         return self.conv(x)
   184: 
   185: 
   186: class MobileNetV2(nn.Module):
   187:     """MobileNetV2 adapted for CIFAR/small-image input (Sandler et al., 2018).
   188: 
   189:     Uses stride-1 initial conv (no stride-2) for 32x32 input.
   190:     Width multiplier = 1.0, ~2.2M parameters.
   191:     """
   192: 
   193:     CFG = [
   194:         # expand_ratio, channels, num_blocks, stride
   195:         [1, 16, 1, 1],
   196:         [6, 24, 2, 1],
   197:         [6, 32, 3, 2],
   198:         [6, 64, 4, 2],
   199:         [6, 96, 3, 1],
   200:         [6, 160, 3, 2],
   201:         [6, 320, 1, 1],
   202:     ]
   203: 
   204:     def __init__(self, num_classes=10):
   205:         super().__init__()
   206:         self.conv1 = nn.Sequential(
   207:             nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
   208:             nn.BatchNorm2d(32),
   209:             nn.ReLU6(inplace=True),
   210:         )
   211:         layers = []
   212:         inp = 32
   213:         for t, c, n, s in self.CFG:
   214:             for i in range(n):
   215:                 stride = s if i == 0 else 1
   216:                 layers.append(InvertedResidual(inp, c, stride, t))
   217:                 inp = c
   218:         self.layers = nn.Sequential(*layers)
   219:         self.conv_last = nn.Sequential(
   220:             nn.Conv2d(320, 1280, 1, bias=False),
   221:             nn.BatchNorm2d(1280),
   222:             nn.ReLU6(inplace=True),
   223:         )
   224:         self.pool = CustomPool()
   225:         self.fc = nn.Linear(1280, num_classes)
   226: 
   227:     def forward(self, x):
   228:         x = self.conv1(x)
   229:         x = self.layers(x)
   230:         x = self.conv_last(x)
   231:         x = self.pool(x)
   232:         return self.fc(x)
   233: 
   234: 
   235: def build_model(arch, num_classes):
   236:     """Build model by architecture name."""
   237:     if arch == 'resnet20':
   238:         return ResNet(BasicBlock, [3, 3, 3], num_classes)
   239:     elif arch == 'resnet56':
   240:         return ResNet(BasicBlock, [9, 9, 9], num_classes)
   241:     elif arch == 'vgg16bn':
   242:         return VGG(num_classes)
   243:     elif arch == 'mobilenetv2':
   244:         return MobileNetV2(num_classes)
   245:     else:
   246:         raise ValueError(f"Unknown architecture: {arch}")
   247: 
   248: 
   249: # ============================================================================
   250: # Weight Initialization (FIXED)
   251: # ============================================================================
   252: 
   253: def initialize_weights(model):
   254:     """Kaiming initialization for all layers."""
   255:     for m in model.modules():
   256:         if isinstance(m, nn.Conv2d):
   257:             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
   258:         elif isinstance(m, nn.BatchNorm2d):
   259:             nn.init.constant_(m.weight, 1)
   260:             nn.init.constant_(m.bias, 0)
   261:         elif isinstance(m, nn.Linear):
   262:             nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
   263:             if m.bias is not None:
   264:                 nn.init.constant_(m.bias, 0)
   265: 
   266: 
   267: # ============================================================================
   268: # Data Loading (FIXED)
   269: # ============================================================================
   270: 
   271: def get_dataloaders(dataset, data_root, batch_size=128, num_workers=4):
   272:     """Create train/test dataloaders with standard augmentation."""
   273:     if dataset == 'cifar10':
   274:         mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
   275:         num_classes = 10
   276:         Dataset = torchvision.datasets.CIFAR10
   277:     elif dataset == 'cifar100':
   278:         mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
   279:         num_classes = 100
   280:         Dataset = torchvision.datasets.CIFAR100
   281:     elif dataset == 'fmnist':
   282:         mean, std = (0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)
   283:         num_classes = 10
   284:         Dataset = torchvision.datasets.FashionMNIST
   285:     else:
   286:         raise ValueError(f"Unknown dataset: {dataset}")
   287: 
   288:     is_grayscale = (dataset == 'fmnist')
   289: 
   290:     train_transform_list = [
   291:         transforms.Resize(32),
   292:         transforms.RandomCrop(32, padding=4),
   293:         transforms.RandomHorizontalFlip(),
   294:         transforms.ToTensor(),
   295:     ]
   296:     if is_grayscale:
   297:         train_transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
   298:     train_transform_list.append(transforms.Normalize(mean, std))
   299:     train_transform = transforms.Compose(train_transform_list)
   300: 
   301:     test_transform_list = [
   302:         transforms.Resize(32),
   303:         transforms.ToTensor(),
   304:     ]
   305:     if is_grayscale:
   306:         test_transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
   307:     test_transform_list.append(transforms.Normalize(mean, std))
   308:     test_transform = transforms.Compose(test_transform_list)
   309: 
   310:     train_set = Dataset(root=data_root, train=True, download=False, transform=train_transform)
   311:     test_set = Dataset(root=data_root, train=False, download=False, transform=test_transform)
   312: 
   313:     train_loader = torch.utils.data.DataLoader(
   314:         train_set, batch_size=batch_size, shuffle=True,
   315:         num_workers=num_workers, pin_memory=True,
   316:     )
   317:     test_loader = torch.utils.data.DataLoader(
   318:         test_set, batch_size=batch_size, shuffle=False,
   319:         num_workers=num_workers, pin_memory=True,
   320:     )
   321:     return train_loader, test_loader, num_classes
   322: 
   323: 
   324: # ============================================================================
   325: # Training Loop (FIXED)
   326: # ============================================================================
   327: 
   328: def train_epoch(model, loader, criterion, optimizer, device):
   329:     """Train for one epoch. Returns (avg_loss, accuracy%)."""
   330:     model.train()
   331:     total_loss, correct, total = 0.0, 0, 0
   332:     for inputs, targets in loader:
   333:         inputs, targets = inputs.to(device), targets.to(device)
   334:         optimizer.zero_grad()
   335:         outputs = model(inputs)
   336:         loss = criterion(outputs, targets)
   337:         loss.backward()
   338:         optimizer.step()
   339:         total_loss += loss.item() * inputs.size(0)
   340:         _, predicted = outputs.max(1)
   341:         correct += predicted.eq(targets).sum().item()
   342:         total += inputs.size(0)
   343:     return total_loss / total, 100.0 * correct / total
   344: 
   345: 
   346: def evaluate(model, loader, criterion, device):
   347:     """Evaluate on test set. Returns (avg_loss, accuracy%)."""
   348:     model.eval()
   349:     total_loss, correct, total = 0.0, 0, 0
   350:     with torch.no_grad():
   351:         for inputs, targets in loader:
   352:             inputs, targets = inputs.to(device), targets.to(device)
   353:             outputs = model(inputs)
   354:             loss = criterion(outputs, targets)
   355:             total_loss += loss.item() * inputs.size(0)
   356:             _, predicted = outputs.max(1)
   357:             correct += predicted.eq(targets).sum().item()
   358:             total += inputs.size(0)
   359:     return total_loss / total, 100.0 * correct / total
   360: 
   361: 
   362: def main():
   363:     parser = argparse.ArgumentParser(description="CV Pooling Aggregation Benchmark")
   364:     parser.add_argument('--arch', type=str, required=True,
   365:                         choices=['resnet20', 'resnet56', 'vgg16bn', 'mobilenetv2'])
   366:     parser.add_argument('--dataset', type=str, required=True,
   367:                         choices=['cifar10', 'cifar100', 'fmnist'])
   368:     parser.add_argument('--data-root', type=str, default='/data/cifar')
   369:     parser.add_argument('--epochs', type=int, default=200)
   370:     parser.add_argument('--batch-size', type=int, default=128)
   371:     parser.add_argument('--lr', type=float, default=0.1)
   372:     parser.add_argument('--momentum', type=float, default=0.9)
   373:     parser.add_argument('--weight-decay', type=float, default=5e-4)
   374:     parser.add_argument('--seed', type=int, default=42)
   375:     parser.add_argument('--output-dir', type=str, default='.')
   376:     args = parser.parse_args()
   377: 
   378:     torch.manual_seed(args.seed)
   379:     torch.cuda.manual_seed_all(args.seed)
   380:     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   381: 
   382:     # Data
   383:     train_loader, test_loader, num_classes = get_dataloaders(
   384:         args.dataset, args.data_root, args.batch_size,
   385:     )
   386: 
   387:     # Model
   388:     model = build_model(args.arch, num_classes)
   389: 
   390:     # Initialize
   391:     initialize_weights(model)
   392:     model = model.to(device)
   393: 
   394:     # Optimizer
   395:     criterion = nn.CrossEntropyLoss()
   396:     optimizer = optim.SGD(
   397:         model.parameters(), lr=args.lr,
   398:         momentum=args.momentum, weight_decay=args.weight_decay,
   399:     )
   400:     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
   401: 
   402:     # Train
   403:     best_acc = 0.0
   404:     for epoch in range(args.epochs):
   405:         train_loss, train_acc = train_epoch(
   406:             model, train_loader, criterion, optimizer, device,
   407:         )
   408:         test_loss, test_acc = evaluate(model, test_loader, criterion, device)
   409:         scheduler.step()
   410: 
   411:         if (epoch + 1) % 10 == 0 or epoch == 0:
   412:             print(
   413:                 f"TRAIN_METRICS: epoch={epoch+1} train_loss={train_loss:.4f} "
   414:                 f"train_acc={train_acc:.2f} test_loss={test_loss:.4f} "
   415:                 f"test_acc={test_acc:.2f} lr={optimizer.param_groups[0]['lr']:.6f}",
   416:                 flush=True,
   417:             )
   418: 
   419:         if test_acc > best_acc:
   420:             best_acc = test_acc
   421: 
   422:     print(f"TEST_METRICS: test_acc={best_acc:.2f}", flush=True)
   423: 
   424: 
   425: if __name__ == '__main__':
   426:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `global_max` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_pool.py`:

```python
Lines 31–42:
    28: # ============================================================================
    29: 
    30: # -- EDITABLE REGION START (lines 31-48) ------------------------------------
    31: class CustomPool(nn.Module):
    32:     """Global Max Pooling.
    33: 
    34:     Selects the maximum activation per channel across spatial dimensions.
    35:     Captures the most salient features rather than averaging over all positions.
    36:     """
    37: 
    38:     def __init__(self):
    39:         super().__init__()
    40: 
    41:     def forward(self, x):
    42:         return F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)
    43: # -- EDITABLE REGION END (lines 31-48) --------------------------------------
    44: 
    45: 
```

### `gem` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_pool.py`:

```python
Lines 31–47:
    28: # ============================================================================
    29: 
    30: # -- EDITABLE REGION START (lines 31-48) ------------------------------------
    31: class CustomPool(nn.Module):
    32:     """Generalized Mean (GeM) Pooling.
    33: 
    34:     Learnable generalized mean with parameter p (init=3.0).
    35:     Interpolates between average pooling (p=1) and max pooling (p->inf).
    36: 
    37:     """
    38: 
    39:     def __init__(self):
    40:         super().__init__()
    41:         self.p = nn.Parameter(torch.ones(1) * 3.0)
    42:         self.eps = 1e-6
    43: 
    44:     def forward(self, x):
    45:         p = self.p.clamp(min=1.0)
    46:         x = x.clamp(min=self.eps)
    47:         return F.adaptive_avg_pool2d(x.pow(p), 1).pow(1.0 / p).view(x.size(0), -1)
    48: # -- EDITABLE REGION END (lines 31-48) --------------------------------------
    49: 
    50: 
```

### `avg_max` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_pool.py`:

```python
Lines 31–44:
    28: # ============================================================================
    29: 
    30: # -- EDITABLE REGION START (lines 31-48) ------------------------------------
    31: class CustomPool(nn.Module):
    32:     """Average + Max Pooling.
    33: 
    34:     Element-wise mean of global average pooling and global max pooling.
    35:     Combines mean-field statistics with peak activations.
    36:     """
    37: 
    38:     def __init__(self):
    39:         super().__init__()
    40: 
    41:     def forward(self, x):
    42:         avg = F.adaptive_avg_pool2d(x, 1)
    43:         mx = F.adaptive_max_pool2d(x, 1)
    44:         return ((avg + mx) / 2).view(x.size(0), -1)
    45: # -- EDITABLE REGION END (lines 31-48) --------------------------------------
    46: 
    47: 
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
