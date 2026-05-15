# MLS-Bench: dl-weight-initialization

# DL Weight Initialization Strategy Design

## Research Question
Design a data-independent weight initialization strategy for deep convolutional neural networks that improves convergence speed and final test accuracy across different architectures and datasets, while keeping the data pipeline, optimizer, schedule, loss, and model definitions fixed.

## Background
Weight initialization is fundamental to training deep networks. Poor initialization leads to vanishing or exploding gradients, slow convergence, or worse generalization. Representative methods include:

- **Kaiming / He initialization** (He et al., "Delving Deep into Rectifiers", ICCV 2015, arXiv:1502.01852): for ReLU-style nonlinearities, draws conv weights from `N(0, sqrt(2 / fan_mode))` (typically `fan_in` or `fan_out`).
- **Orthogonal initialization** (Saxe, McClelland & Ganguli, ICLR 2014, arXiv:1312.6120): preserves signal norms via random orthogonal matrices, motivated by the dynamics of deep linear networks.
- **Fixup** (Zhang, Dauphin & Ma, ICLR 2019, arXiv:1901.09321): for residual networks without normalization. Scales the last conv in each residual block by `L^(-1/(2m-2))` where `L` is the number of residual blocks and `m` is the number of conv layers per block (commonly `2`); zero-initializes the last conv per block so residual branches start near identity; adds learnable scalar biases / multipliers around each conv.
- **Zero / near-zero residual init** (subset of Fixup-style ideas): zero-initialize the last weight in each residual branch.
- **LSUV** (Mishkin & Matas, 2015): orthogonal init followed by per-layer rescaling of activation variance to `1` using a small calibration batch.

Each of these addresses one aspect of initialization; there is room for strategies that jointly account for residual structure, BatchNorm's rescaling effect, depth, and the interaction between conv and classifier layers.

## What You Can Modify
The `initialize_weights(model, config)` function inside `pytorch-vision/custom_init.py`. The function receives the fully constructed model and a `config` dict, and must initialize all parameters in place.

`config` provides:
- `arch` (str)
- `num_classes` (int)
- `depth` (int): number of `Conv2d` + `Linear` layers in the model.

You may iterate over `model.named_modules()` or `model.named_parameters()` and design per-layer or depth-dependent strategies, treat residual shortcut projections separately from main-path convs, set `BatchNorm2d` weight/bias differently, and use any data-independent logic. No access to training data and no calibration passes.

## Fixed Pipeline
- Optimizer: SGD with `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`.
- Schedule: cosine annealing over `200` epochs.
- Data augmentation: `RandomCrop(32, pad=4)` + `RandomHorizontalFlip`.
- Evaluation settings: ResNet-56 on CIFAR-100, VGG-16-BN on CIFAR-100, MobileNetV2 on FashionMNIST.

## Baselines
- **kaiming_normal** — He et al., arXiv:1502.01852; conv weights from `N(0, sqrt(2/fan_out))`, zero biases, BatchNorm `(weight=1, bias=0)`.
- **fixup** — Zhang et al., arXiv:1901.09321; scales the first residual conv by `L^(-1/(2m-2))` with `m=2` and zero-initializes the last conv per residual block.
- **orthogonal** — Saxe et al., arXiv:1312.6120; orthogonal init for conv and linear layers (gain `sqrt(2)` for ReLU), zero biases, BatchNorm `(weight=1, bias=0)`.

## Metric
Best test accuracy (%, higher is better) achieved during training. The initialization must be data-independent and must not run calibration passes, alter the model graph, change optimizer hyperparameters, or modify evaluation behavior.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/pytorch-vision/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `pytorch-vision/custom_init.py`
- editable lines **228–261**




## Readable Context


### `pytorch-vision/custom_init.py`  [EDITABLE — lines 228–261 only]

```python
     1: """CV Weight Initialization Benchmark.
     2: 
     3: Train vision models (ResNet, VGG, MobileNetV2) on CIFAR-10/100/FashionMNIST to evaluate
     4: weight initialization strategies.
     5: 
     6: FIXED: Model architectures, data pipeline, training loop.
     7: EDITABLE: initialize_weights() function.
     8: 
     9: Usage:
    10:     python custom_init.py --arch resnet20 --dataset cifar10 --seed 42
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
    27: # Model Architectures (FIXED)
    28: # ============================================================================
    29: 
    30: class BasicBlock(nn.Module):
    31:     """Basic residual block for CIFAR ResNets."""
    32:     expansion = 1
    33: 
    34:     def __init__(self, in_planes, planes, stride=1):
    35:         super().__init__()
    36:         self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
    37:         self.bn1 = nn.BatchNorm2d(planes)
    38:         self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
    39:         self.bn2 = nn.BatchNorm2d(planes)
    40:         self.shortcut = nn.Sequential()
    41:         if stride != 1 or in_planes != planes * self.expansion:
    42:             self.shortcut = nn.Sequential(
    43:                 nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
    44:                 nn.BatchNorm2d(planes * self.expansion),
    45:             )
    46: 
    47:     def forward(self, x):
    48:         out = F.relu(self.bn1(self.conv1(x)))
    49:         out = self.bn2(self.conv2(out))
    50:         out += self.shortcut(x)
    51:         return F.relu(out)
    52: 
    53: 
    54: class ResNet(nn.Module):
    55:     """CIFAR-adapted ResNet (He et al., 2016).
    56: 
    57:     Uses 3x3 initial conv (no 7x7), no max pooling, global avg pool at end.
    58:     Standard depths: ResNet-20 ([3,3,3]), ResNet-56 ([9,9,9]).
    59:     """
    60: 
    61:     def __init__(self, block, num_blocks, num_classes=10):
    62:         super().__init__()
    63:         self.in_planes = 16
    64:         self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
    65:         self.bn1 = nn.BatchNorm2d(16)
    66:         self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
    67:         self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
    68:         self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
    69:         self.fc = nn.Linear(64 * block.expansion, num_classes)
    70: 
    71:     def _make_layer(self, block, planes, num_blocks, stride):
    72:         strides = [stride] + [1] * (num_blocks - 1)
    73:         layers = []
    74:         for s in strides:
    75:             layers.append(block(self.in_planes, planes, s))
    76:             self.in_planes = planes * block.expansion
    77:         return nn.Sequential(*layers)
    78: 
    79:     def forward(self, x):
    80:         out = F.relu(self.bn1(self.conv1(x)))
    81:         out = self.layer1(out)
    82:         out = self.layer2(out)
    83:         out = self.layer3(out)
    84:         out = F.adaptive_avg_pool2d(out, 1)
    85:         out = out.view(out.size(0), -1)
    86:         return self.fc(out)
    87: 
    88: 
    89: class VGG(nn.Module):
    90:     """VGG-16 with BatchNorm, adapted for CIFAR (Simonyan & Zisserman, 2015).
    91: 
    92:     Uses adaptive avg pool instead of large FC layers, suitable for 32x32 input.
    93:     """
    94: 
    95:     VGG16_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
    96:                  512, 512, 512, 'M', 512, 512, 512, 'M']
    97: 
    98:     def __init__(self, num_classes=100):
    99:         super().__init__()
   100:         self.features = self._make_layers(self.VGG16_CFG)
   101:         self.classifier = nn.Sequential(
   102:             nn.Linear(512, 512),
   103:             nn.ReLU(True),
   104:             nn.Dropout(0.5),
   105:             nn.Linear(512, num_classes),
   106:         )
   107: 
   108:     def _make_layers(self, cfg):
   109:         layers = []
   110:         in_channels = 3
   111:         for v in cfg:
   112:             if v == 'M':
   113:                 layers.append(nn.MaxPool2d(2, 2))
   114:             else:
   115:                 layers += [
   116:                     nn.Conv2d(in_channels, v, 3, padding=1),
   117:                     nn.BatchNorm2d(v),
   118:                     nn.ReLU(inplace=True),
   119:                 ]
   120:                 in_channels = v
   121:         return nn.Sequential(*layers)
   122: 
   123:     def forward(self, x):
   124:         x = self.features(x)
   125:         x = F.adaptive_avg_pool2d(x, 1)
   126:         x = x.view(x.size(0), -1)
   127:         return self.classifier(x)
   128: 
   129: 
   130: class InvertedResidual(nn.Module):
   131:     """MobileNetV2 inverted residual block (Sandler et al., 2018)."""
   132: 
   133:     def __init__(self, inp, oup, stride, expand_ratio):
   134:         super().__init__()
   135:         self.stride = stride
   136:         hidden = int(round(inp * expand_ratio))
   137:         self.use_res = (stride == 1 and inp == oup)
   138:         layers = []
   139:         if expand_ratio != 1:
   140:             layers += [
   141:                 nn.Conv2d(inp, hidden, 1, bias=False),
   142:                 nn.BatchNorm2d(hidden),
   143:                 nn.ReLU6(inplace=True),
   144:             ]
   145:         layers += [
   146:             nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
   147:             nn.BatchNorm2d(hidden),
   148:             nn.ReLU6(inplace=True),
   149:             nn.Conv2d(hidden, oup, 1, bias=False),
   150:             nn.BatchNorm2d(oup),
   151:         ]
   152:         self.conv = nn.Sequential(*layers)
   153: 
   154:     def forward(self, x):
   155:         if self.use_res:
   156:             return x + self.conv(x)
   157:         return self.conv(x)
   158: 
   159: 
   160: class MobileNetV2(nn.Module):
   161:     """MobileNetV2 adapted for CIFAR/small-image input (Sandler et al., 2018).
   162: 
   163:     Uses stride-1 initial conv (no stride-2) for 32x32 input.
   164:     Width multiplier = 1.0, ~2.2M parameters.
   165:     """
   166: 
   167:     CFG = [
   168:         # expand_ratio, channels, num_blocks, stride
   169:         [1, 16, 1, 1],
   170:         [6, 24, 2, 1],
   171:         [6, 32, 3, 2],
   172:         [6, 64, 4, 2],
   173:         [6, 96, 3, 1],
   174:         [6, 160, 3, 2],
   175:         [6, 320, 1, 1],
   176:     ]
   177: 
   178:     def __init__(self, num_classes=10):
   179:         super().__init__()
   180:         self.conv1 = nn.Sequential(
   181:             nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
   182:             nn.BatchNorm2d(32),
   183:             nn.ReLU6(inplace=True),
   184:         )
   185:         layers = []
   186:         inp = 32
   187:         for t, c, n, s in self.CFG:
   188:             for i in range(n):
   189:                 stride = s if i == 0 else 1
   190:                 layers.append(InvertedResidual(inp, c, stride, t))
   191:                 inp = c
   192:         self.layers = nn.Sequential(*layers)
   193:         self.conv_last = nn.Sequential(
   194:             nn.Conv2d(320, 1280, 1, bias=False),
   195:             nn.BatchNorm2d(1280),
   196:             nn.ReLU6(inplace=True),
   197:         )
   198:         self.fc = nn.Linear(1280, num_classes)
   199: 
   200:     def forward(self, x):
   201:         x = self.conv1(x)
   202:         x = self.layers(x)
   203:         x = self.conv_last(x)
   204:         x = F.adaptive_avg_pool2d(x, 1)
   205:         x = x.view(x.size(0), -1)
   206:         return self.fc(x)
   207: 
   208: 
   209: def build_model(arch, num_classes):
   210:     """Build model by architecture name."""
   211:     if arch == 'resnet20':
   212:         return ResNet(BasicBlock, [3, 3, 3], num_classes)
   213:     elif arch == 'resnet56':
   214:         return ResNet(BasicBlock, [9, 9, 9], num_classes)
   215:     elif arch == 'vgg16bn':
   216:         return VGG(num_classes)
   217:     elif arch == 'mobilenetv2':
   218:         return MobileNetV2(num_classes)
   219:     else:
   220:         raise ValueError(f"Unknown architecture: {arch}")
   221: 
   222: 
   223: # ============================================================================
   224: # Weight Initialization
   225: # ============================================================================
   226: 
   227: # -- EDITABLE REGION START (lines 228-261) ------------------------------------
   228: def initialize_weights(model, config):
   229:     """Initialize all weights in the vision model.
   230: 
   231:     Called after model construction, before moving to GPU.
   232: 
   233:     Args:
   234:         model: nn.Module (ResNet or VGG, with Conv2d + BatchNorm2d + Linear)
   235:         config: dict with keys:
   236:             - arch: str ('resnet20', 'resnet56', 'vgg16bn', 'mobilenetv2')
   237:             - num_classes: int (10 or 100)
   238:             - depth: int (total number of Conv2d + Linear layers)
   239: 
   240:     Layer types in the model:
   241:         - nn.Conv2d: feature extractors (no bias when followed by BN)
   242:         - nn.BatchNorm2d: batch normalization (weight=scale, bias=shift)
   243:         - nn.Linear: classifier head (with bias)
   244: 
   245:     Design considerations:
   246:         - Fan-in vs fan-out scaling for Conv2d
   247:         - Distribution choice (normal, uniform, orthogonal, ...)
   248:         - Depth-dependent scaling for residual branches
   249:         - BatchNorm parameter initialization
   250:         - Interaction between initialization and residual shortcuts
   251:     """
   252:     for m in model.modules():
   253:         if isinstance(m, nn.Conv2d):
   254:             nn.init.normal_(m.weight, 0, 0.01)
   255:         elif isinstance(m, nn.BatchNorm2d):
   256:             nn.init.constant_(m.weight, 1)
   257:             nn.init.constant_(m.bias, 0)
   258:         elif isinstance(m, nn.Linear):
   259:             nn.init.normal_(m.weight, 0, 0.01)
   260:             if m.bias is not None:
   261:                 nn.init.constant_(m.bias, 0)
   262: # -- EDITABLE REGION END (lines 228-261) --------------------------------------
   263: 
   264: 
   265: # ============================================================================
   266: # Data Loading (FIXED)
   267: # ============================================================================
   268: 
   269: def get_dataloaders(dataset, data_root, batch_size=128, num_workers=4):
   270:     """Create train/test dataloaders with standard augmentation."""
   271:     if dataset == 'cifar10':
   272:         mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
   273:         num_classes = 10
   274:         Dataset = torchvision.datasets.CIFAR10
   275:     elif dataset == 'cifar100':
   276:         mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
   277:         num_classes = 100
   278:         Dataset = torchvision.datasets.CIFAR100
   279:     elif dataset == 'fmnist':
   280:         mean, std = (0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)
   281:         num_classes = 10
   282:         Dataset = torchvision.datasets.FashionMNIST
   283:     else:
   284:         raise ValueError(f"Unknown dataset: {dataset}")
   285: 
   286:     is_grayscale = (dataset == 'fmnist')
   287: 
   288:     train_transform_list = [
   289:         transforms.Resize(32),
   290:         transforms.RandomCrop(32, padding=4),
   291:         transforms.RandomHorizontalFlip(),
   292:         transforms.ToTensor(),
   293:     ]
   294:     if is_grayscale:
   295:         train_transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
   296:     train_transform_list.append(transforms.Normalize(mean, std))
   297:     train_transform = transforms.Compose(train_transform_list)
   298: 
   299:     test_transform_list = [
   300:         transforms.Resize(32),
   301:         transforms.ToTensor(),
   302:     ]
   303:     if is_grayscale:
   304:         test_transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
   305:     test_transform_list.append(transforms.Normalize(mean, std))
   306:     test_transform = transforms.Compose(test_transform_list)
   307: 
   308:     train_set = Dataset(root=data_root, train=True, download=False, transform=train_transform)
   309:     test_set = Dataset(root=data_root, train=False, download=False, transform=test_transform)
   310: 
   311:     train_loader = torch.utils.data.DataLoader(
   312:         train_set, batch_size=batch_size, shuffle=True,
   313:         num_workers=num_workers, pin_memory=True,
   314:     )
   315:     test_loader = torch.utils.data.DataLoader(
   316:         test_set, batch_size=batch_size, shuffle=False,
   317:         num_workers=num_workers, pin_memory=True,
   318:     )
   319:     return train_loader, test_loader, num_classes
   320: 
   321: 
   322: # ============================================================================
   323: # Training Loop (FIXED)
   324: # ============================================================================
   325: 
   326: def train_epoch(model, loader, criterion, optimizer, device):
   327:     """Train for one epoch. Returns (avg_loss, accuracy%)."""
   328:     model.train()
   329:     total_loss, correct, total = 0.0, 0, 0
   330:     for inputs, targets in loader:
   331:         inputs, targets = inputs.to(device), targets.to(device)
   332:         optimizer.zero_grad()
   333:         outputs = model(inputs)
   334:         loss = criterion(outputs, targets)
   335:         loss.backward()
   336:         optimizer.step()
   337:         total_loss += loss.item() * inputs.size(0)
   338:         _, predicted = outputs.max(1)
   339:         correct += predicted.eq(targets).sum().item()
   340:         total += inputs.size(0)
   341:     return total_loss / total, 100.0 * correct / total
   342: 
   343: 
   344: def evaluate(model, loader, criterion, device):
   345:     """Evaluate on test set. Returns (avg_loss, accuracy%)."""
   346:     model.eval()
   347:     total_loss, correct, total = 0.0, 0, 0
   348:     with torch.no_grad():
   349:         for inputs, targets in loader:
   350:             inputs, targets = inputs.to(device), targets.to(device)
   351:             outputs = model(inputs)
   352:             loss = criterion(outputs, targets)
   353:             total_loss += loss.item() * inputs.size(0)
   354:             _, predicted = outputs.max(1)
   355:             correct += predicted.eq(targets).sum().item()
   356:             total += inputs.size(0)
   357:     return total_loss / total, 100.0 * correct / total
   358: 
   359: 
   360: def main():
   361:     parser = argparse.ArgumentParser(description="CV Weight Initialization Benchmark")
   362:     parser.add_argument('--arch', type=str, required=True,
   363:                         choices=['resnet20', 'resnet56', 'vgg16bn', 'mobilenetv2'])
   364:     parser.add_argument('--dataset', type=str, required=True,
   365:                         choices=['cifar10', 'cifar100', 'fmnist'])
   366:     parser.add_argument('--data-root', type=str, default='/data/cifar')
   367:     parser.add_argument('--epochs', type=int, default=200)
   368:     parser.add_argument('--batch-size', type=int, default=128)
   369:     parser.add_argument('--lr', type=float, default=0.1)
   370:     parser.add_argument('--momentum', type=float, default=0.9)
   371:     parser.add_argument('--weight-decay', type=float, default=5e-4)
   372:     parser.add_argument('--seed', type=int, default=42)
   373:     parser.add_argument('--output-dir', type=str, default='.')
   374:     args = parser.parse_args()
   375: 
   376:     torch.manual_seed(args.seed)
   377:     torch.cuda.manual_seed_all(args.seed)
   378:     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   379: 
   380:     # Data
   381:     train_loader, test_loader, num_classes = get_dataloaders(
   382:         args.dataset, args.data_root, args.batch_size,
   383:     )
   384: 
   385:     # Model
   386:     model = build_model(args.arch, num_classes)
   387:     depth = sum(1 for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear)))
   388:     config = {'arch': args.arch, 'num_classes': num_classes, 'depth': depth}
   389: 
   390:     # Initialize
   391:     initialize_weights(model, config)
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




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **resnet56-cifar100** — wall-clock budget `00:59:00`, compute share `1.0`
- **vgg16bn-cifar100** — wall-clock budget `00:59:00`, compute share `1.0`
- **mobilenetv2-fmnist** — wall-clock budget `00:59:00`, compute share `1.0`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `kaiming_normal` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_init.py`:

```python
Lines 228–244:
   225: # ============================================================================
   226: 
   227: # -- EDITABLE REGION START (lines 228-261) ------------------------------------
   228: def initialize_weights(model, config):
   229:     """Kaiming/He normal initialization (fan_out, ReLU).
   230: 
   231:     Conv2d: N(0, sqrt(2/fan_out)) — preserves forward-pass variance with ReLU.
   232:     BatchNorm2d: weight=1, bias=0.
   233:     Linear: N(0, sqrt(2/fan_in)).
   234:     """
   235:     for m in model.modules():
   236:         if isinstance(m, nn.Conv2d):
   237:             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
   238:         elif isinstance(m, nn.BatchNorm2d):
   239:             nn.init.constant_(m.weight, 1)
   240:             nn.init.constant_(m.bias, 0)
   241:         elif isinstance(m, nn.Linear):
   242:             nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
   243:             if m.bias is not None:
   244:                 nn.init.constant_(m.bias, 0)
   245: # -- EDITABLE REGION END (lines 228-261) --------------------------------------
   246: 
   247: 
```

### `fixup` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_init.py`:

```python
Lines 228–261:
   225: # ============================================================================
   226: 
   227: # -- EDITABLE REGION START (lines 228-261) ------------------------------------
   228: def initialize_weights(model, config):
   229:     """Fixup-inspired residual scaling with zero-gamma BatchNorm.
   230: 
   231:     For ResNets: Kaiming normal for all Conv2d, then scale the last conv in
   232:     each residual block by n_blocks^(-0.5) to control variance accumulation.
   233:     Zero-initialize the last BN in each block (Goyal et al., 2017).
   234:     For VGG: Kaiming normal (no residual branches to scale).
   235:     Linear: small normal init with zero bias.
   236:     """
   237:     arch = config['arch']
   238:     is_resnet = arch.startswith('resnet')
   239: 
   240:     # Phase 1: standard Kaiming init for all layers
   241:     for m in model.modules():
   242:         if isinstance(m, nn.Conv2d):
   243:             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
   244:         elif isinstance(m, nn.BatchNorm2d):
   245:             nn.init.constant_(m.weight, 1)
   246:             nn.init.constant_(m.bias, 0)
   247:         elif isinstance(m, nn.Linear):
   248:             nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
   249:             if m.bias is not None:
   250:                 nn.init.constant_(m.bias, 0)
   251: 
   252:     # Phase 2: Fixup-inspired residual branch scaling for ResNets
   253:     if is_resnet:
   254:         n_blocks = sum(1 for m in model.modules() if isinstance(m, BasicBlock))
   255:         fixup_scale = n_blocks ** (-0.5)
   256:         for m in model.modules():
   257:             if isinstance(m, BasicBlock):
   258:                 # Scale the last conv (conv2) in each residual block
   259:                 m.conv2.weight.data.mul_(fixup_scale)
   260:                 # Zero-init the last BN so residual branch starts near identity
   261:                 nn.init.constant_(m.bn2.weight, 0)
   262: # -- EDITABLE REGION END (lines 228-261) --------------------------------------
   263: 
   264: 
```

### `orthogonal` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_init.py`:

```python
Lines 228–244:
   225: # ============================================================================
   226: 
   227: # -- EDITABLE REGION START (lines 228-261) ------------------------------------
   228: def initialize_weights(model, config):
   229:     """Orthogonal initialization.
   230: 
   231:     Conv2d & Linear: orthogonal matrix (gain=sqrt(2) for ReLU).
   232:     BatchNorm2d: weight=1, bias=0.
   233:     """
   234:     gain = nn.init.calculate_gain('relu')
   235:     for m in model.modules():
   236:         if isinstance(m, nn.Conv2d):
   237:             nn.init.orthogonal_(m.weight, gain=gain)
   238:         elif isinstance(m, nn.BatchNorm2d):
   239:             nn.init.constant_(m.weight, 1)
   240:             nn.init.constant_(m.bias, 0)
   241:         elif isinstance(m, nn.Linear):
   242:             nn.init.orthogonal_(m.weight, gain=gain)
   243:             if m.bias is not None:
   244:                 nn.init.constant_(m.bias, 0)
   245: # -- EDITABLE REGION END (lines 228-261) --------------------------------------
   246: 
   247: 
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
