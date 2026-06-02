# MLS-Bench: dl-lr-schedule

# DL Learning Rate Schedule Design

## Research Question
Design a learning-rate schedule for training deep convolutional image classifiers that improves convergence speed and final test accuracy across different architectures and datasets, while keeping the optimizer type, training loop, and all other hyperparameters fixed.

## Background
Learning-rate scheduling is critical for training deep networks effectively: a fixed learning rate is often too high (unstable) or too low (slow). Representative schedules include:

- **Step decay** (He et al., "Deep Residual Learning for Image Recognition", arXiv:1512.03385): divide the learning rate by 10 at fixed milestones (e.g. 50% and 75% of training).
- **Cosine annealing** (Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts", ICLR 2017, arXiv:1608.03983): smooth decay following a cosine curve from `base_lr` down to a small final value (often `0`).
- **Warmup + cosine** (Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour", arXiv:1706.02677): linear warmup over a small number of epochs followed by cosine (or step) decay; stabilizes large-batch / high learning rates.
- **One-Cycle / Super-convergence** (Smith & Topin, "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates", arXiv:1708.07120): a single triangular ramp up then ramp down, often combined with momentum cycling.
- **Polynomial decay**: `lr(t) = base_lr * (1 - t/T)^p`, common in segmentation literature.

These schedules are usually designed without considering architecture-specific properties (depth, residual structure, BatchNorm) or dataset characteristics; there is room for schedules that adapt to context.

## What You Can Modify
The `get_lr(epoch, total_epochs, base_lr, config)` function inside `pytorch-vision/custom_schedule.py`. The function is called once per epoch and must return the learning rate (a float) used by SGD for that epoch.

`config` provides:
- `arch` (str: e.g. `'resnet20'`, `'resnet56'`, `'mobilenetv2'`)
- `dataset` (str: e.g. `'cifar10'`, `'cifar100'`, `'fmnist'`)

You may freely shape the LR curve (cosine, polynomial, exponential, linear, piecewise), include warmup of arbitrary length and shape, set any minimum/final LR, condition on `arch` and `dataset`, and use any epoch-dependent logic such as cyclic restarts, sharp transitions, or plateaus.

## Fixed Pipeline
- Optimizer: SGD with `lr=base_lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`. The task setup uses **no** built-in PyTorch scheduler — your `get_lr` directly determines the per-epoch learning rate.
- Training: `200` epochs.
- Data augmentation: `RandomCrop(32, pad=4)` + `RandomHorizontalFlip`.
- Weight initialization: Kaiming normal (fixed, not editable).
- Evaluation settings: ResNet-20 on CIFAR-10, ResNet-56 on CIFAR-100, MobileNetV2 on FashionMNIST.

## Baselines
- **cosine** — Loshchilov & Hutter, arXiv:1608.03983; standard cosine annealing from `base_lr` to `0` over `total_epochs`.
- **warmup_cosine** — Goyal et al., arXiv:1706.02677; linear warmup (commonly 5 epochs) followed by cosine annealing to `0`.
- **one_cycle** — Smith & Topin, arXiv:1708.07120; triangular up-then-down ramp that peaks above `base_lr` and ends below it.

## Metric
Best test accuracy (%, higher is better) achieved during training. The schedule must not modify model code, data augmentation, loss functions, optimizer type, weight decay, or evaluation.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/pytorch-vision/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `pytorch-vision/custom_schedule.py`
- editable lines **246–269**




## Readable Context


### `pytorch-vision/custom_schedule.py`  [EDITABLE — lines 246–269 only]

```python
     1: """CV Learning Rate Schedule Benchmark.
     2: 
     3: Train vision models (ResNet, VGG, MobileNetV2) on CIFAR-10/100/FashionMNIST to evaluate
     4: learning rate schedule strategies.
     5: 
     6: FIXED: Model architectures, data pipeline, training loop, optimizer.
     7: EDITABLE: get_lr() function.
     8: 
     9: Usage:
    10:     python custom_schedule.py --arch resnet20 --dataset cifar10 --seed 42
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
   224: # Weight Initialization (FIXED)
   225: # ============================================================================
   226: 
   227: def initialize_weights(model):
   228:     """Kaiming normal initialization for all layers."""
   229:     for m in model.modules():
   230:         if isinstance(m, nn.Conv2d):
   231:             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
   232:         elif isinstance(m, nn.BatchNorm2d):
   233:             nn.init.constant_(m.weight, 1)
   234:             nn.init.constant_(m.bias, 0)
   235:         elif isinstance(m, nn.Linear):
   236:             nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
   237:             if m.bias is not None:
   238:                 nn.init.constant_(m.bias, 0)
   239: 
   240: 
   241: # ============================================================================
   242: # Learning Rate Schedule
   243: # ============================================================================
   244: 
   245: # -- EDITABLE REGION START (lines 246-269) ------------------------------------
   246: def get_lr(epoch, total_epochs, base_lr, config):
   247:     """Compute learning rate for the given epoch.
   248: 
   249:     Called once per epoch to set the learning rate for all parameter groups.
   250: 
   251:     Args:
   252:         epoch: current epoch (0-indexed, ranges from 0 to total_epochs-1)
   253:         total_epochs: total number of training epochs
   254:         base_lr: initial learning rate (from --lr flag, default 0.1)
   255:         config: dict with keys:
   256:             - arch: str ('resnet20', 'resnet56', 'vgg16bn', 'mobilenetv2')
   257:             - dataset: str ('cifar10', 'cifar100', 'fmnist')
   258: 
   259:     Returns:
   260:         float: learning rate to use for this epoch
   261: 
   262:     Design considerations:
   263:         - Warmup phase to stabilize early training
   264:         - Decay shape (step, cosine, polynomial, exponential, ...)
   265:         - Final learning rate (decay to zero vs small constant)
   266:         - Architecture/dataset-aware scheduling
   267:         - Interaction with momentum and weight decay
   268:     """
   269:     return base_lr  # constant LR (no schedule)
   270: # -- EDITABLE REGION END (lines 246-269) --------------------------------------
   271: 
   272: 
   273: # ============================================================================
   274: # Data Loading (FIXED)
   275: # ============================================================================
   276: 
   277: def get_dataloaders(dataset, data_root, batch_size=128, num_workers=4):
   278:     """Create train/test dataloaders with standard augmentation."""
   279:     if dataset == 'cifar10':
   280:         mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
   281:         num_classes = 10
   282:         Dataset = torchvision.datasets.CIFAR10
   283:     elif dataset == 'cifar100':
   284:         mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
   285:         num_classes = 100
   286:         Dataset = torchvision.datasets.CIFAR100
   287:     elif dataset == 'fmnist':
   288:         mean, std = (0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)
   289:         num_classes = 10
   290:         Dataset = torchvision.datasets.FashionMNIST
   291:     else:
   292:         raise ValueError(f"Unknown dataset: {dataset}")
   293: 
   294:     is_grayscale = (dataset == 'fmnist')
   295: 
   296:     train_transform_list = [
   297:         transforms.Resize(32),
   298:         transforms.RandomCrop(32, padding=4),
   299:         transforms.RandomHorizontalFlip(),
   300:         transforms.ToTensor(),
   301:     ]
   302:     if is_grayscale:
   303:         train_transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
   304:     train_transform_list.append(transforms.Normalize(mean, std))
   305:     train_transform = transforms.Compose(train_transform_list)
   306: 
   307:     test_transform_list = [
   308:         transforms.Resize(32),
   309:         transforms.ToTensor(),
   310:     ]
   311:     if is_grayscale:
   312:         test_transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
   313:     test_transform_list.append(transforms.Normalize(mean, std))
   314:     test_transform = transforms.Compose(test_transform_list)
   315: 
   316:     train_set = Dataset(root=data_root, train=True, download=False, transform=train_transform)
   317:     test_set = Dataset(root=data_root, train=False, download=False, transform=test_transform)
   318: 
   319:     train_loader = torch.utils.data.DataLoader(
   320:         train_set, batch_size=batch_size, shuffle=True,
   321:         num_workers=num_workers, pin_memory=True,
   322:     )
   323:     test_loader = torch.utils.data.DataLoader(
   324:         test_set, batch_size=batch_size, shuffle=False,
   325:         num_workers=num_workers, pin_memory=True,
   326:     )
   327:     return train_loader, test_loader, num_classes
   328: 
   329: 
   330: # ============================================================================
   331: # Training Loop (FIXED)
   332: # ============================================================================
   333: 
   334: def train_epoch(model, loader, criterion, optimizer, device):
   335:     """Train for one epoch. Returns (avg_loss, accuracy%)."""
   336:     model.train()
   337:     total_loss, correct, total = 0.0, 0, 0
   338:     for inputs, targets in loader:
   339:         inputs, targets = inputs.to(device), targets.to(device)
   340:         optimizer.zero_grad()
   341:         outputs = model(inputs)
   342:         loss = criterion(outputs, targets)
   343:         loss.backward()
   344:         optimizer.step()
   345:         total_loss += loss.item() * inputs.size(0)
   346:         _, predicted = outputs.max(1)
   347:         correct += predicted.eq(targets).sum().item()
   348:         total += inputs.size(0)
   349:     return total_loss / total, 100.0 * correct / total
   350: 
   351: 
   352: def evaluate(model, loader, criterion, device):
   353:     """Evaluate on test set. Returns (avg_loss, accuracy%)."""
   354:     model.eval()
   355:     total_loss, correct, total = 0.0, 0, 0
   356:     with torch.no_grad():
   357:         for inputs, targets in loader:
   358:             inputs, targets = inputs.to(device), targets.to(device)
   359:             outputs = model(inputs)
   360:             loss = criterion(outputs, targets)
   361:             total_loss += loss.item() * inputs.size(0)
   362:             _, predicted = outputs.max(1)
   363:             correct += predicted.eq(targets).sum().item()
   364:             total += inputs.size(0)
   365:     return total_loss / total, 100.0 * correct / total
   366: 
   367: 
   368: def main():
   369:     parser = argparse.ArgumentParser(description="CV Learning Rate Schedule Benchmark")
   370:     parser.add_argument('--arch', type=str, required=True,
   371:                         choices=['resnet20', 'resnet56', 'vgg16bn', 'mobilenetv2'])
   372:     parser.add_argument('--dataset', type=str, required=True,
   373:                         choices=['cifar10', 'cifar100', 'fmnist'])
   374:     parser.add_argument('--data-root', type=str, default='/data/cifar')
   375:     parser.add_argument('--epochs', type=int, default=200)
   376:     parser.add_argument('--batch-size', type=int, default=128)
   377:     parser.add_argument('--lr', type=float, default=0.1)
   378:     parser.add_argument('--momentum', type=float, default=0.9)
   379:     parser.add_argument('--weight-decay', type=float, default=5e-4)
   380:     parser.add_argument('--seed', type=int, default=42)
   381:     parser.add_argument('--output-dir', type=str, default='.')
   382:     args = parser.parse_args()
   383: 
   384:     torch.manual_seed(args.seed)
   385:     torch.cuda.manual_seed_all(args.seed)
   386:     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   387: 
   388:     # Data
   389:     train_loader, test_loader, num_classes = get_dataloaders(
   390:         args.dataset, args.data_root, args.batch_size,
   391:     )
   392: 
   393:     # Model
   394:     model = build_model(args.arch, num_classes)
   395:     initialize_weights(model)
   396:     model = model.to(device)
   397: 
   398:     # Optimizer -- plain SGD, NO scheduler (LR set manually via get_lr)
   399:     criterion = nn.CrossEntropyLoss()
   400:     optimizer = optim.SGD(
   401:         model.parameters(), lr=args.lr,
   402:         momentum=args.momentum, weight_decay=args.weight_decay,
   403:     )
   404: 
   405:     # Config for get_lr
   406:     config = {'arch': args.arch, 'dataset': args.dataset}
   407: 
   408:     # Train
   409:     best_acc = 0.0
   410:     for epoch in range(args.epochs):
   411:         # Set learning rate manually each epoch
   412:         lr = get_lr(epoch, args.epochs, args.lr, config)
   413:         for pg in optimizer.param_groups:
   414:             pg['lr'] = lr
   415: 
   416:         train_loss, train_acc = train_epoch(
   417:             model, train_loader, criterion, optimizer, device,
   418:         )
   419:         test_loss, test_acc = evaluate(model, test_loader, criterion, device)
   420: 
   421:         if (epoch + 1) % 10 == 0 or epoch == 0:
   422:             print(
   423:                 f"TRAIN_METRICS: epoch={epoch+1} train_loss={train_loss:.4f} "
   424:                 f"train_acc={train_acc:.2f} test_loss={test_loss:.4f} "
   425:                 f"test_acc={test_acc:.2f} lr={lr:.6f}",
   426:                 flush=True,
   427:             )
   428: 
   429:         if test_acc > best_acc:
   430:             best_acc = test_acc
   431: 
   432:     print(f"TEST_METRICS: test_acc={best_acc:.2f}", flush=True)
   433: 
   434: 
   435: if __name__ == '__main__':
   436:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `cosine` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_schedule.py`:

```python
Lines 246–251:
   243: # ============================================================================
   244: 
   245: # -- EDITABLE REGION START (lines 246-269) ------------------------------------
   246: def get_lr(epoch, total_epochs, base_lr, config):
   247:     """Cosine annealing from base_lr to 0.
   248: 
   249:     LR = base_lr * 0.5 * (1 + cos(pi * epoch / total_epochs))
   250:     """
   251:     return base_lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
   252: # -- EDITABLE REGION END (lines 246-269) --------------------------------------
   253: 
   254: 
```

### `warmup_cosine` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_schedule.py`:

```python
Lines 246–256:
   243: # ============================================================================
   244: 
   245: # -- EDITABLE REGION START (lines 246-269) ------------------------------------
   246: def get_lr(epoch, total_epochs, base_lr, config):
   247:     """Linear warmup (5 epochs) then cosine decay to 0.
   248: 
   249:     Epochs 0-4: linearly ramp from base_lr/5 to base_lr.
   250:     Epochs 5+: cosine anneal from base_lr to 0.
   251:     """
   252:     warmup = 5
   253:     if epoch < warmup:
   254:         return base_lr * (epoch + 1) / warmup
   255:     progress = (epoch - warmup) / (total_epochs - warmup)
   256:     return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
   257: # -- EDITABLE REGION END (lines 246-269) --------------------------------------
   258: 
   259: 
```

### `one_cycle` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_schedule.py`:

```python
Lines 246–268:
   243: # ============================================================================
   244: 
   245: # -- EDITABLE REGION START (lines 246-269) ------------------------------------
   246: def get_lr(epoch, total_epochs, base_lr, config):
   247:     """OneCycleLR schedule (Smith & Topin, 2019).
   248: 
   249:     Phase 1 (0-30%): cosine warmup from base_lr/25 to base_lr.
   250:     Phase 2 (30-100%): cosine anneal from base_lr to base_lr/25.
   251:     """
   252:     pct_start = 0.3
   253:     div_factor = 25.0
   254:     final_div = 25.0
   255: 
   256:     min_lr = base_lr / div_factor
   257:     final_lr = base_lr / final_div
   258: 
   259:     progress = epoch / max(total_epochs - 1, 1)
   260: 
   261:     if progress <= pct_start:
   262:         # Warmup phase: cosine from min_lr to base_lr
   263:         t = progress / pct_start
   264:         return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * (1 - t)))
   265:     else:
   266:         # Anneal phase: cosine from base_lr to final_lr
   267:         t = (progress - pct_start) / (1 - pct_start)
   268:         return final_lr + (base_lr - final_lr) * 0.5 * (1 + math.cos(math.pi * t))
   269: # -- EDITABLE REGION END (lines 246-269) --------------------------------------
   270: 
   271: 
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
