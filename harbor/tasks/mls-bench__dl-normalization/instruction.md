# MLS-Bench: dl-normalization

# DL Normalization Layer Design

## Research Question
Design a normalization layer for deep convolutional neural networks that improves training stability and final test accuracy across different architectures and datasets, while keeping the optimizer, data pipeline, and outer training loop fixed.

## Background
Normalization layers are critical in modern deep networks: they control activation scale, mitigate internal covariate shift, and enable stable training at higher learning rates. Representative methods include:

- **BatchNorm** (Ioffe & Szegedy, ICML 2015, arXiv:1502.03167): normalizes across the batch dimension per channel; the de facto standard, but depends on batch statistics and behaves differently at train and test time.
- **GroupNorm** (Wu & He, ECCV 2018, arXiv:1803.08494): divides channels into groups and normalizes within each group; batch-size independent.
- **InstanceNorm** (Ulyanov, Vedaldi & Lempitsky, "Instance Normalization: The Missing Ingredient for Fast Stylization", arXiv:1607.08022): normalizes each channel independently per instance; common in style transfer.
- **LayerNorm** (Ba, Kiros & Hinton, arXiv:1607.06450): normalizes across all channels for each sample; standard in transformers.
- **RMSNorm** (Zhang & Sennrich, NeurIPS 2019, arXiv:1910.07467): normalizes by root-mean-square only (no mean centering); cheaper than LayerNorm.
- **Batch-Instance Norm (BIN)** (Nam & Kim, NeurIPS 2018, arXiv:1805.07925): per-channel learnable mixture of BatchNorm and InstanceNorm.
- **Switchable Normalization (SN)** (Luo et al., "Differentiable Learning-to-Normalize via Switchable Normalization", arXiv:1806.10779): learnable convex combination of BN/LN/IN statistics per layer.
- **EvoNorm** (Liu et al., "Evolving Normalization-Activation Layers", NeurIPS 2020, arXiv:2004.02967): jointly evolves normalization and activation.

Each method has limitations: BatchNorm degrades with small batches, GroupNorm requires choosing the number of groups, InstanceNorm discards inter-channel information, and LayerNorm may not suit spatial feature maps. There is room for designs that combine strengths or use novel normalization statistics.

## What You Can Modify
The `CustomNorm` class inside `pytorch-vision/custom_norm.py`. It must be a drop-in replacement for `nn.BatchNorm2d`:

- Constructor: `CustomNorm(num_features)` where `num_features` is the channel count `C`.
- Input shape: `[B, C, H, W]`. Output shape: `[B, C, H, W]`.
- Train and eval behavior must be numerically stable.

You may modify normalization statistics (mean/variance over batch, channel, spatial, or any combination), learnable affine parameters (scale and shift), grouping strategies, mixtures of normalization approaches, and adaptive or input-dependent normalization, as long as the interface is preserved.

## Fixed Pipeline
The optimizer, learning-rate schedule, data augmentation, backbones, activations, datasets, and loss functions are fixed by the harness.

The normalization module must preserve tensor shape, accept the expected channel count, remain numerically stable in train and eval, and must not change backbones, activations, datasets, loss functions, or optimizer settings.

## Baselines
- **group_norm** — Wu & He, arXiv:1803.08494; default `num_groups=32` (paper-recommended), with channel counts smaller than `num_groups` falling back to InstanceNorm-equivalent grouping.
- **batch_instance_norm** — Nam & Kim, arXiv:1805.07925; per-channel learnable gate `rho` initialized to `1.0` (BatchNorm-leaning, matching the paper).
- **switchable_norm** — Luo et al., arXiv:1806.10779; learnable softmax weights over `{BN, LN, IN}` statistics per layer.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/pytorch-vision/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `pytorch-vision/custom_norm.py`
- editable lines **31–45**




## Readable Context


### `pytorch-vision/custom_norm.py`  [EDITABLE — lines 31–45 only]

```python
     1: """CV Normalization Layer Benchmark.
     2: 
     3: Train vision models (ResNet, VGG, MobileNetV2) on CIFAR-10/100/FashionMNIST to evaluate
     4: normalization layer designs.
     5: 
     6: FIXED: Model architectures, data pipeline, training loop.
     7: EDITABLE: CustomNorm class.
     8: 
     9: Usage:
    10:     python custom_norm.py --arch resnet20 --dataset cifar10 --seed 42
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
    27: # Normalization Layer
    28: # ============================================================================
    29: 
    30: # -- EDITABLE REGION START (lines 31-45) --------------------------------------
    31: class CustomNorm(nn.Module):
    32:     """Custom normalization layer. Drop-in replacement for BatchNorm2d.
    33: 
    34:     Args:
    35:         num_features: number of channels C
    36:     Input: [B, C, H, W]
    37:     Output: [B, C, H, W]
    38:     """
    39: 
    40:     def __init__(self, num_features):
    41:         super().__init__()
    42:         self.norm = nn.BatchNorm2d(num_features)
    43: 
    44:     def forward(self, x):
    45:         return self.norm(x)
    46: # -- EDITABLE REGION END (lines 31-45) ----------------------------------------
    47: 
    48: 
    49: # ============================================================================
    50: # Model Architectures (FIXED)
    51: # ============================================================================
    52: 
    53: class BasicBlock(nn.Module):
    54:     """Basic residual block for CIFAR ResNets."""
    55:     expansion = 1
    56: 
    57:     def __init__(self, in_planes, planes, stride=1):
    58:         super().__init__()
    59:         self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
    60:         self.bn1 = CustomNorm(planes)
    61:         self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
    62:         self.bn2 = CustomNorm(planes)
    63:         self.shortcut = nn.Sequential()
    64:         if stride != 1 or in_planes != planes * self.expansion:
    65:             self.shortcut = nn.Sequential(
    66:                 nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
    67:                 CustomNorm(planes * self.expansion),
    68:             )
    69: 
    70:     def forward(self, x):
    71:         out = F.relu(self.bn1(self.conv1(x)))
    72:         out = self.bn2(self.conv2(out))
    73:         out += self.shortcut(x)
    74:         return F.relu(out)
    75: 
    76: 
    77: class ResNet(nn.Module):
    78:     """CIFAR-adapted ResNet (He et al., 2016).
    79: 
    80:     Uses 3x3 initial conv (no 7x7), no max pooling, global avg pool at end.
    81:     Standard depths: ResNet-20 ([3,3,3]), ResNet-56 ([9,9,9]), ResNet-110 ([18,18,18]).
    82:     """
    83: 
    84:     def __init__(self, block, num_blocks, num_classes=10):
    85:         super().__init__()
    86:         self.in_planes = 16
    87:         self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
    88:         self.bn1 = CustomNorm(16)
    89:         self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
    90:         self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
    91:         self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
    92:         self.fc = nn.Linear(64 * block.expansion, num_classes)
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
   109:         return self.fc(out)
   110: 
   111: 
   112: class VGG(nn.Module):
   113:     """VGG-16 with BatchNorm, adapted for CIFAR (Simonyan & Zisserman, 2015).
   114: 
   115:     Uses adaptive avg pool instead of large FC layers, suitable for 32x32 input.
   116:     """
   117: 
   118:     VGG16_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
   119:                  512, 512, 512, 'M', 512, 512, 512, 'M']
   120: 
   121:     def __init__(self, num_classes=100):
   122:         super().__init__()
   123:         self.features = self._make_layers(self.VGG16_CFG)
   124:         self.classifier = nn.Sequential(
   125:             nn.Linear(512, 512),
   126:             nn.ReLU(True),
   127:             nn.Dropout(0.5),
   128:             nn.Linear(512, num_classes),
   129:         )
   130: 
   131:     def _make_layers(self, cfg):
   132:         layers = []
   133:         in_channels = 3
   134:         for v in cfg:
   135:             if v == 'M':
   136:                 layers.append(nn.MaxPool2d(2, 2))
   137:             else:
   138:                 layers += [
   139:                     nn.Conv2d(in_channels, v, 3, padding=1),
   140:                     CustomNorm(v),
   141:                     nn.ReLU(inplace=True),
   142:                 ]
   143:                 in_channels = v
   144:         return nn.Sequential(*layers)
   145: 
   146:     def forward(self, x):
   147:         x = self.features(x)
   148:         x = F.adaptive_avg_pool2d(x, 1)
   149:         x = x.view(x.size(0), -1)
   150:         return self.classifier(x)
   151: 
   152: 
   153: class InvertedResidual(nn.Module):
   154:     """MobileNetV2 inverted residual block (Sandler et al., 2018)."""
   155: 
   156:     def __init__(self, inp, oup, stride, expand_ratio):
   157:         super().__init__()
   158:         self.stride = stride
   159:         hidden = int(round(inp * expand_ratio))
   160:         self.use_res = (stride == 1 and inp == oup)
   161:         layers = []
   162:         if expand_ratio != 1:
   163:             layers += [
   164:                 nn.Conv2d(inp, hidden, 1, bias=False),
   165:                 CustomNorm(hidden),
   166:                 nn.ReLU6(inplace=True),
   167:             ]
   168:         layers += [
   169:             nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
   170:             CustomNorm(hidden),
   171:             nn.ReLU6(inplace=True),
   172:             nn.Conv2d(hidden, oup, 1, bias=False),
   173:             CustomNorm(oup),
   174:         ]
   175:         self.conv = nn.Sequential(*layers)
   176: 
   177:     def forward(self, x):
   178:         if self.use_res:
   179:             return x + self.conv(x)
   180:         return self.conv(x)
   181: 
   182: 
   183: class MobileNetV2(nn.Module):
   184:     """MobileNetV2 adapted for CIFAR/small-image input (Sandler et al., 2018).
   185: 
   186:     Uses stride-1 initial conv (no stride-2) for 32x32 input.
   187:     Width multiplier = 1.0, ~2.2M parameters.
   188:     """
   189: 
   190:     CFG = [
   191:         # expand_ratio, channels, num_blocks, stride
   192:         [1, 16, 1, 1],
   193:         [6, 24, 2, 1],
   194:         [6, 32, 3, 2],
   195:         [6, 64, 4, 2],
   196:         [6, 96, 3, 1],
   197:         [6, 160, 3, 2],
   198:         [6, 320, 1, 1],
   199:     ]
   200: 
   201:     def __init__(self, num_classes=10):
   202:         super().__init__()
   203:         self.conv1 = nn.Sequential(
   204:             nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
   205:             CustomNorm(32),
   206:             nn.ReLU6(inplace=True),
   207:         )
   208:         layers = []
   209:         inp = 32
   210:         for t, c, n, s in self.CFG:
   211:             for i in range(n):
   212:                 stride = s if i == 0 else 1
   213:                 layers.append(InvertedResidual(inp, c, stride, t))
   214:                 inp = c
   215:         self.layers = nn.Sequential(*layers)
   216:         self.conv_last = nn.Sequential(
   217:             nn.Conv2d(320, 1280, 1, bias=False),
   218:             CustomNorm(1280),
   219:             nn.ReLU6(inplace=True),
   220:         )
   221:         self.fc = nn.Linear(1280, num_classes)
   222: 
   223:     def forward(self, x):
   224:         x = self.conv1(x)
   225:         x = self.layers(x)
   226:         x = self.conv_last(x)
   227:         x = F.adaptive_avg_pool2d(x, 1)
   228:         x = x.view(x.size(0), -1)
   229:         return self.fc(x)
   230: 
   231: 
   232: def build_model(arch, num_classes):
   233:     """Build model by architecture name."""
   234:     if arch == 'resnet20':
   235:         return ResNet(BasicBlock, [3, 3, 3], num_classes)
   236:     elif arch == 'resnet56':
   237:         return ResNet(BasicBlock, [9, 9, 9], num_classes)
   238:     elif arch == 'resnet110':
   239:         return ResNet(BasicBlock, [18, 18, 18], num_classes)
   240:     elif arch == 'vgg16bn':
   241:         return VGG(num_classes)
   242:     elif arch == 'mobilenetv2':
   243:         return MobileNetV2(num_classes)
   244:     else:
   245:         raise ValueError(f"Unknown architecture: {arch}")
   246: 
   247: 
   248: # ============================================================================
   249: # Weight Initialization (FIXED)
   250: # ============================================================================
   251: 
   252: def initialize_weights(model):
   253:     """Initialize all weights in the vision model."""
   254:     for m in model.modules():
   255:         if isinstance(m, nn.Conv2d):
   256:             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
   257:         elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d, nn.LayerNorm)):
   258:             if hasattr(m, 'weight') and m.weight is not None:
   259:                 nn.init.constant_(m.weight, 1)
   260:             if hasattr(m, 'bias') and m.bias is not None:
   261:                 nn.init.constant_(m.bias, 0)
   262:         elif isinstance(m, nn.Linear):
   263:             nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
   264:             if m.bias is not None:
   265:                 nn.init.constant_(m.bias, 0)
   266:         elif hasattr(m, 'weight') and m.weight is not None and m.weight.dim() == 1:
   267:             nn.init.constant_(m.weight, 1)
   268:             if hasattr(m, 'bias') and m.bias is not None:
   269:                 nn.init.constant_(m.bias, 0)
   270: 
   271: 
   272: # ============================================================================
   273: # Data Loading (FIXED)
   274: # ============================================================================
   275: 
   276: def get_dataloaders(dataset, data_root, batch_size=128, num_workers=4):
   277:     """Create train/test dataloaders with standard augmentation."""
   278:     if dataset == 'cifar10':
   279:         mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
   280:         num_classes = 10
   281:         Dataset = torchvision.datasets.CIFAR10
   282:     elif dataset == 'cifar100':
   283:         mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
   284:         num_classes = 100
   285:         Dataset = torchvision.datasets.CIFAR100
   286:     elif dataset == 'fmnist':
   287:         mean, std = (0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)
   288:         num_classes = 10
   289:         Dataset = torchvision.datasets.FashionMNIST
   290:     else:
   291:         raise ValueError(f"Unknown dataset: {dataset}")
   292: 
   293:     is_grayscale = (dataset == 'fmnist')
   294: 
   295:     train_transform_list = [
   296:         transforms.Resize(32),
   297:         transforms.RandomCrop(32, padding=4),
   298:         transforms.RandomHorizontalFlip(),
   299:         transforms.ToTensor(),
   300:     ]
   301:     if is_grayscale:
   302:         train_transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
   303:     train_transform_list.append(transforms.Normalize(mean, std))
   304:     train_transform = transforms.Compose(train_transform_list)
   305: 
   306:     test_transform_list = [
   307:         transforms.Resize(32),
   308:         transforms.ToTensor(),
   309:     ]
   310:     if is_grayscale:
   311:         test_transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
   312:     test_transform_list.append(transforms.Normalize(mean, std))
   313:     test_transform = transforms.Compose(test_transform_list)
   314: 
   315:     train_set = Dataset(root=data_root, train=True, download=False, transform=train_transform)
   316:     test_set = Dataset(root=data_root, train=False, download=False, transform=test_transform)
   317: 
   318:     train_loader = torch.utils.data.DataLoader(
   319:         train_set, batch_size=batch_size, shuffle=True,
   320:         num_workers=num_workers, pin_memory=True,
   321:     )
   322:     test_loader = torch.utils.data.DataLoader(
   323:         test_set, batch_size=batch_size, shuffle=False,
   324:         num_workers=num_workers, pin_memory=True,
   325:     )
   326:     return train_loader, test_loader, num_classes
   327: 
   328: 
   329: # ============================================================================
   330: # Training Loop (FIXED)
   331: # ============================================================================
   332: 
   333: def train_epoch(model, loader, criterion, optimizer, device):
   334:     """Train for one epoch. Returns (avg_loss, accuracy%)."""
   335:     model.train()
   336:     total_loss, correct, total = 0.0, 0, 0
   337:     for inputs, targets in loader:
   338:         inputs, targets = inputs.to(device), targets.to(device)
   339:         optimizer.zero_grad()
   340:         outputs = model(inputs)
   341:         loss = criterion(outputs, targets)
   342:         loss.backward()
   343:         optimizer.step()
   344:         total_loss += loss.item() * inputs.size(0)
   345:         _, predicted = outputs.max(1)
   346:         correct += predicted.eq(targets).sum().item()
   347:         total += inputs.size(0)
   348:     return total_loss / total, 100.0 * correct / total
   349: 
   350: 
   351: def evaluate(model, loader, criterion, device):
   352:     """Evaluate on test set. Returns (avg_loss, accuracy%)."""
   353:     model.eval()
   354:     total_loss, correct, total = 0.0, 0, 0
   355:     with torch.no_grad():
   356:         for inputs, targets in loader:
   357:             inputs, targets = inputs.to(device), targets.to(device)
   358:             outputs = model(inputs)
   359:             loss = criterion(outputs, targets)
   360:             total_loss += loss.item() * inputs.size(0)
   361:             _, predicted = outputs.max(1)
   362:             correct += predicted.eq(targets).sum().item()
   363:             total += inputs.size(0)
   364:     return total_loss / total, 100.0 * correct / total
   365: 
   366: 
   367: def main():
   368:     parser = argparse.ArgumentParser(description="CV Normalization Layer Benchmark")
   369:     parser.add_argument('--arch', type=str, required=True,
   370:                         choices=['resnet20', 'resnet56', 'resnet110', 'vgg16bn', 'mobilenetv2'])
   371:     parser.add_argument('--dataset', type=str, required=True,
   372:                         choices=['cifar10', 'cifar100', 'fmnist'])
   373:     parser.add_argument('--data-root', type=str, default='/data/cifar')
   374:     parser.add_argument('--epochs', type=int, default=200)
   375:     parser.add_argument('--batch-size', type=int, default=128)
   376:     parser.add_argument('--lr', type=float, default=0.1)
   377:     parser.add_argument('--momentum', type=float, default=0.9)
   378:     parser.add_argument('--weight-decay', type=float, default=5e-4)
   379:     parser.add_argument('--seed', type=int, default=42)
   380:     parser.add_argument('--output-dir', type=str, default='.')
   381:     args = parser.parse_args()
   382: 
   383:     torch.manual_seed(args.seed)
   384:     torch.cuda.manual_seed_all(args.seed)
   385:     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   386: 
   387:     # Data
   388:     train_loader, test_loader, num_classes = get_dataloaders(
   389:         args.dataset, args.data_root, args.batch_size,
   390:     )
   391: 
   392:     # Model
   393:     model = build_model(args.arch, num_classes)
   394: 
   395:     # Initialize
   396:     initialize_weights(model)
   397:     model = model.to(device)
   398: 
   399:     # Optimizer
   400:     criterion = nn.CrossEntropyLoss()
   401:     optimizer = optim.SGD(
   402:         model.parameters(), lr=args.lr,
   403:         momentum=args.momentum, weight_decay=args.weight_decay,
   404:     )
   405:     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
   406: 
   407:     # Train
   408:     best_acc = 0.0
   409:     for epoch in range(args.epochs):
   410:         train_loss, train_acc = train_epoch(
   411:             model, train_loader, criterion, optimizer, device,
   412:         )
   413:         test_loss, test_acc = evaluate(model, test_loader, criterion, device)
   414:         scheduler.step()
   415: 
   416:         if (epoch + 1) % 10 == 0 or epoch == 0:
   417:             print(
   418:                 f"TRAIN_METRICS: epoch={epoch+1} train_loss={train_loss:.4f} "
   419:                 f"train_acc={train_acc:.2f} test_loss={test_loss:.4f} "
   420:                 f"test_acc={test_acc:.2f} lr={optimizer.param_groups[0]['lr']:.6f}",
   421:                 flush=True,
   422:             )
   423: 
   424:         if test_acc > best_acc:
   425:             best_acc = test_acc
   426: 
   427:     print(f"TEST_METRICS: test_acc={best_acc:.2f}", flush=True)
   428: 
   429: 
   430: if __name__ == '__main__':
   431:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `group_norm` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_norm.py`:

```python
Lines 31–49:
    28: # ============================================================================
    29: 
    30: # -- EDITABLE REGION START (lines 31-45) --------------------------------------
    31: class CustomNorm(nn.Module):
    32:     """Group Normalization for 2D feature maps. Drop-in replacement for BatchNorm2d.
    33: 
    34:     Divides channels into groups and normalizes within each group independently.
    35:     Works well with small batch sizes where BatchNorm statistics are noisy.
    36: 
    37:     Reference: Wu & He, "Group Normalization" (ECCV 2018)
    38:     """
    39: 
    40:     def __init__(self, num_features):
    41:         super().__init__()
    42:         num_groups = min(32, num_features)
    43:         # Ensure num_features is divisible by num_groups
    44:         while num_features % num_groups != 0:
    45:             num_groups -= 1
    46:         self.norm = nn.GroupNorm(num_groups, num_features)
    47: 
    48:     def forward(self, x):
    49:         return self.norm(x)
    50: # -- EDITABLE REGION END (lines 31-45) ----------------------------------------
    51: 
    52: 
```

### `batch_instance_norm` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_norm.py`:

```python
Lines 31–64:
    28: # ============================================================================
    29: 
    30: # -- EDITABLE REGION START (lines 31-45) --------------------------------------
    31: class CustomNorm(nn.Module):
    32:     """Batch-Instance Normalization for 2D feature maps. Drop-in replacement for BatchNorm2d.
    33: 
    34:     Learns a per-channel gate rho in [0, 1] (via sigmoid) that interpolates
    35:     between BatchNorm statistics and InstanceNorm statistics.
    36: 
    37:     Reference: Nam & Kim, "Batch-Instance Normalization for Adaptively
    38:     Style-Invariant Neural Networks" (NeurIPS 2018)
    39:     """
    40: 
    41:     def __init__(self, num_features):
    42:         super().__init__()
    43:         self.num_features = num_features
    44:         self.eps = 1e-5
    45:         # Learnable affine parameters
    46:         self.weight = nn.Parameter(torch.ones(num_features))
    47:         self.bias = nn.Parameter(torch.zeros(num_features))
    48:         # Gate parameter (before sigmoid); init at 1.0 -> sigmoid ~ 0.73 -> mostly BN
    49:         self.rho = nn.Parameter(torch.ones(num_features) * 1.0)
    50: 
    51:     def forward(self, x):
    52:         # x: [B, C, H, W]
    53:         gate = torch.sigmoid(self.rho).view(1, -1, 1, 1)
    54:         # Batch stats: per C over (B, H, W)
    55:         mean_bn = x.mean(dim=(0, 2, 3), keepdim=True)
    56:         var_bn = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
    57:         # Instance stats: per (B, C) over (H, W)
    58:         mean_in = x.mean(dim=(2, 3), keepdim=True)
    59:         var_in = x.var(dim=(2, 3), keepdim=True, unbiased=False)
    60:         # Interpolate
    61:         x_bn = (x - mean_bn) / (var_bn + self.eps).sqrt()
    62:         x_in = (x - mean_in) / (var_in + self.eps).sqrt()
    63:         x_norm = gate * x_bn + (1 - gate) * x_in
    64:         return x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
    65: # -- EDITABLE REGION END (lines 31-45) ----------------------------------------
    66: 
    67: 
```

### `switchable_norm` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_norm.py`:

```python
Lines 31–72:
    28: # ============================================================================
    29: 
    30: # -- EDITABLE REGION START (lines 31-45) --------------------------------------
    31: class CustomNorm(nn.Module):
    32:     """Switchable Normalization for 2D feature maps. Drop-in replacement for BatchNorm2d.
    33: 
    34:     Learns to combine BatchNorm, InstanceNorm, and LayerNorm statistics via
    35:     softmax-weighted importance weights. Adapts normalization strategy per
    36:     channel during training.
    37: 
    38:     Reference: Luo et al., "Differentiable Learning-to-Normalize via
    39:     Switchable Normalization" (ICLR 2019)
    40:     """
    41: 
    42:     def __init__(self, num_features):
    43:         super().__init__()
    44:         self.num_features = num_features
    45:         self.eps = 1e-5
    46:         # Learnable affine parameters
    47:         self.weight = nn.Parameter(torch.ones(num_features))
    48:         self.bias = nn.Parameter(torch.zeros(num_features))
    49:         # Importance weights for mean (3 norms) and var (3 norms)
    50:         self.mean_weight = nn.Parameter(torch.ones(3))
    51:         self.var_weight = nn.Parameter(torch.ones(3))
    52: 
    53:     def forward(self, x):
    54:         # x: [B, C, H, W]
    55:         B, C, H, W = x.shape
    56:         # Softmax over importance weights
    57:         mean_w = F.softmax(self.mean_weight, dim=0)
    58:         var_w = F.softmax(self.var_weight, dim=0)
    59:         # Instance stats: per (B, C) over (H, W)
    60:         mean_in = x.mean(dim=(2, 3), keepdim=True)
    61:         var_in = x.var(dim=(2, 3), keepdim=True, unbiased=False)
    62:         # Layer stats: per B over (C, H, W)
    63:         mean_ln = x.mean(dim=(1, 2, 3), keepdim=True)
    64:         var_ln = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
    65:         # Batch stats: per C over (B, H, W)
    66:         mean_bn = x.mean(dim=(0, 2, 3), keepdim=True)
    67:         var_bn = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
    68:         # Weighted combination
    69:         mean = mean_w[0] * mean_in + mean_w[1] * mean_ln + mean_w[2] * mean_bn
    70:         var = var_w[0] * var_in + var_w[1] * var_ln + var_w[2] * var_bn
    71:         x_norm = (x - mean) / (var + self.eps).sqrt()
    72:         return x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
    73: # -- EDITABLE REGION END (lines 31-45) ----------------------------------------
    74: 
    75: 
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
