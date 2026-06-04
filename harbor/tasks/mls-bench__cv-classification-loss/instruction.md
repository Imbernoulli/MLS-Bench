# MLS-Bench: cv-classification-loss

# CV Classification Loss Function Design

## Research Question
Design a classification loss function for deep convolutional neural networks that improves test accuracy across different architectures and datasets, while keeping the model architectures, optimizer, data pipeline, and evaluation loss fixed.

## Background
Cross-entropy is the standard training objective for image classifiers, but it has known limitations: it treats all misclassifications equally, drives confident predictions toward extreme logits without an explicit margin, and does not adapt to training dynamics or class-count differences. Several alternative formulations have been proposed:

- **Label Smoothing** (Szegedy et al., "Rethinking the Inception Architecture for Computer Vision", arXiv:1512.00567): replaces one-hot targets with `(1 - eps) * one_hot + eps / C` to discourage overconfidence.
- **Focal Loss** (Lin et al., ICCV 2017, arXiv:1708.02002): multiplies the per-example cross-entropy by `(1 - p_t)^gamma`, down-weighting easy examples.
- **PolyLoss** (Leng et al., ICLR 2022, arXiv:2204.12511): expresses CE as a polynomial series in `(1 - p_t)` and adds a leading correction term, e.g. `Poly-1 = CE + eps * (1 - p_t)`.

These methods are largely static or address a single failure mode. Possible directions include confidence calibration, epoch-dependent curricula, class-count-aware weighting, learned temperature scaling, or compositions of these ideas.

## What You Can Modify
The `compute_loss(logits, targets, config)` function inside `pytorch-vision/custom_loss.py`. The function receives raw logits `[B, C]`, integer targets `[B]`, and a `config` dict, and must return a differentiable scalar loss.

`config` provides:
- `num_classes` (int)
- `epoch` (int, 0-indexed)
- `total_epochs` (int)

You may use any combination of cross-entropy variants, margin losses, confidence-based reweighting, epoch-dependent curricula, class-count-dependent terms, temperature/logit scaling, or auxiliary regularization (e.g. entropy or logit penalties), as long as the result is a differentiable scalar tensor.

The evaluation loss reported during training (`test_loss`) is computed with standard cross-entropy regardless of the custom loss; the custom loss only affects training.

## Fixed Pipeline
The model architectures, optimizer, learning-rate schedule, data pipeline, augmentation, and the evaluation loss are fixed by the harness and not editable. The custom loss affects training only.

## Baselines
The included baselines provide reference implementations of:
- **label_smoothing** — Szegedy et al., arXiv:1512.00567.
- **focal_loss** — Lin et al., arXiv:1708.02002.
- **poly_loss** — Leng et al., arXiv:2204.12511, Poly-1 form.

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/pytorch-vision/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `pytorch-vision/custom_loss.py`
- editable lines **246–266**




## Readable Context


### `pytorch-vision/custom_loss.py`  [EDITABLE — lines 246–266 only]

```python
     1: """CV Classification Loss Benchmark.
     2: 
     3: Train vision models (ResNet, VGG, MobileNetV2) on CIFAR-10/100/FashionMNIST to evaluate
     4: classification loss function designs.
     5: 
     6: FIXED: Model architectures, weight initialization, data pipeline, training loop.
     7: EDITABLE: compute_loss() function.
     8: 
     9: Usage:
    10:     python custom_loss.py --arch resnet20 --dataset cifar10 --seed 42
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
   224: # Weight Initialization (FIXED -- Kaiming Normal)
   225: # ============================================================================
   226: 
   227: def initialize_weights(model):
   228:     """Kaiming normal initialization (fixed)."""
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
   242: # Classification Loss (EDITABLE)
   243: # ============================================================================
   244: 
   245: # -- EDITABLE REGION START (lines 246-266) ------------------------------------
   246: def compute_loss(logits, targets, config):
   247:     """Compute classification loss.
   248: 
   249:     Args:
   250:         logits: [B, C] raw model output (pre-softmax)
   251:         targets: [B] integer class labels (0 to C-1)
   252:         config: dict with keys:
   253:             - num_classes: int (10 or 100)
   254:             - epoch: int (current epoch, 0-indexed)
   255:             - total_epochs: int (total number of epochs)
   256:     Returns:
   257:         scalar loss tensor (differentiable)
   258: 
   259:     Design considerations:
   260:         - Hard vs soft target distributions
   261:         - Class-frequency or confidence-based reweighting
   262:         - Curriculum or epoch-dependent loss shaping
   263:         - Regularization terms (entropy, margin, logit penalties)
   264:         - Interaction with softmax temperature
   265:     """
   266:     return F.cross_entropy(logits, targets)
   267: # -- EDITABLE REGION END (lines 246-266) --------------------------------------
   268: 
   269: 
   270: # ============================================================================
   271: # Data Loading (FIXED)
   272: # ============================================================================
   273: 
   274: def get_dataloaders(dataset, data_root, batch_size=128, num_workers=4):
   275:     """Create train/test dataloaders with standard augmentation."""
   276:     if dataset == 'cifar10':
   277:         mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
   278:         num_classes = 10
   279:         Dataset = torchvision.datasets.CIFAR10
   280:     elif dataset == 'cifar100':
   281:         mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
   282:         num_classes = 100
   283:         Dataset = torchvision.datasets.CIFAR100
   284:     elif dataset == 'fmnist':
   285:         mean, std = (0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)
   286:         num_classes = 10
   287:         Dataset = torchvision.datasets.FashionMNIST
   288:     else:
   289:         raise ValueError(f"Unknown dataset: {dataset}")
   290: 
   291:     is_grayscale = (dataset == 'fmnist')
   292: 
   293:     train_transform_list = [
   294:         transforms.Resize(32),
   295:         transforms.RandomCrop(32, padding=4),
   296:         transforms.RandomHorizontalFlip(),
   297:         transforms.ToTensor(),
   298:     ]
   299:     if is_grayscale:
   300:         train_transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
   301:     train_transform_list.append(transforms.Normalize(mean, std))
   302:     train_transform = transforms.Compose(train_transform_list)
   303: 
   304:     test_transform_list = [
   305:         transforms.Resize(32),
   306:         transforms.ToTensor(),
   307:     ]
   308:     if is_grayscale:
   309:         test_transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
   310:     test_transform_list.append(transforms.Normalize(mean, std))
   311:     test_transform = transforms.Compose(test_transform_list)
   312: 
   313:     train_set = Dataset(root=data_root, train=True, download=False, transform=train_transform)
   314:     test_set = Dataset(root=data_root, train=False, download=False, transform=test_transform)
   315: 
   316:     train_loader = torch.utils.data.DataLoader(
   317:         train_set, batch_size=batch_size, shuffle=True,
   318:         num_workers=num_workers, pin_memory=True,
   319:     )
   320:     test_loader = torch.utils.data.DataLoader(
   321:         test_set, batch_size=batch_size, shuffle=False,
   322:         num_workers=num_workers, pin_memory=True,
   323:     )
   324:     return train_loader, test_loader, num_classes
   325: 
   326: 
   327: # ============================================================================
   328: # Training Loop (FIXED)
   329: # ============================================================================
   330: 
   331: def train_epoch(model, loader, optimizer, device, config):
   332:     """Train for one epoch. Returns (avg_loss, accuracy%)."""
   333:     model.train()
   334:     total_loss, correct, total = 0.0, 0, 0
   335:     for inputs, targets in loader:
   336:         inputs, targets = inputs.to(device), targets.to(device)
   337:         optimizer.zero_grad()
   338:         outputs = model(inputs)
   339:         loss = compute_loss(outputs, targets, config)
   340:         loss.backward()
   341:         optimizer.step()
   342:         total_loss += loss.item() * inputs.size(0)
   343:         _, predicted = outputs.max(1)
   344:         correct += predicted.eq(targets).sum().item()
   345:         total += inputs.size(0)
   346:     return total_loss / total, 100.0 * correct / total
   347: 
   348: 
   349: def evaluate(model, loader, device, config):
   350:     """Evaluate on test set. Returns (avg_loss, accuracy%)."""
   351:     model.eval()
   352:     total_loss, correct, total = 0.0, 0, 0
   353:     with torch.no_grad():
   354:         for inputs, targets in loader:
   355:             inputs, targets = inputs.to(device), targets.to(device)
   356:             outputs = model(inputs)
   357:             loss = F.cross_entropy(outputs, targets)
   358:             total_loss += loss.item() * inputs.size(0)
   359:             _, predicted = outputs.max(1)
   360:             correct += predicted.eq(targets).sum().item()
   361:             total += inputs.size(0)
   362:     return total_loss / total, 100.0 * correct / total
   363: 
   364: 
   365: def main():
   366:     parser = argparse.ArgumentParser(description="CV Classification Loss Benchmark")
   367:     parser.add_argument('--arch', type=str, required=True,
   368:                         choices=['resnet20', 'resnet56', 'vgg16bn', 'mobilenetv2'])
   369:     parser.add_argument('--dataset', type=str, required=True,
   370:                         choices=['cifar10', 'cifar100', 'fmnist'])
   371:     parser.add_argument('--data-root', type=str, default='/data/cifar')
   372:     parser.add_argument('--epochs', type=int, default=200)
   373:     parser.add_argument('--batch-size', type=int, default=128)
   374:     parser.add_argument('--lr', type=float, default=0.1)
   375:     parser.add_argument('--momentum', type=float, default=0.9)
   376:     parser.add_argument('--weight-decay', type=float, default=5e-4)
   377:     parser.add_argument('--seed', type=int, default=42)
   378:     parser.add_argument('--output-dir', type=str, default='.')
   379:     args = parser.parse_args()
   380: 
   381:     torch.manual_seed(args.seed)
   382:     torch.cuda.manual_seed_all(args.seed)
   383:     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   384: 
   385:     # Data
   386:     train_loader, test_loader, num_classes = get_dataloaders(
   387:         args.dataset, args.data_root, args.batch_size,
   388:     )
   389: 
   390:     # Model
   391:     model = build_model(args.arch, num_classes)
   392: 
   393:     # Initialize (fixed Kaiming normal)
   394:     initialize_weights(model)
   395:     model = model.to(device)
   396: 
   397:     # Optimizer
   398:     optimizer = optim.SGD(
   399:         model.parameters(), lr=args.lr,
   400:         momentum=args.momentum, weight_decay=args.weight_decay,
   401:     )
   402:     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
   403: 
   404:     # Loss config
   405:     loss_config = {
   406:         'num_classes': num_classes,
   407:         'epoch': 0,
   408:         'total_epochs': args.epochs,
   409:     }
   410: 
   411:     # Train
   412:     best_acc = 0.0
   413:     for epoch in range(args.epochs):
   414:         loss_config['epoch'] = epoch
   415:         train_loss, train_acc = train_epoch(
   416:             model, train_loader, optimizer, device, loss_config,
   417:         )
   418:         test_loss, test_acc = evaluate(model, test_loader, device, loss_config)
   419:         scheduler.step()
   420: 
   421:         if (epoch + 1) % 10 == 0 or epoch == 0:
   422:             print(
   423:                 f"TRAIN_METRICS: epoch={epoch+1} train_loss={train_loss:.4f} "
   424:                 f"train_acc={train_acc:.2f} test_loss={test_loss:.4f} "
   425:                 f"test_acc={test_acc:.2f} lr={optimizer.param_groups[0]['lr']:.6f}",
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


### `label_smoothing` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_loss.py`:

```python
Lines 246–252:
   243: # ============================================================================
   244: 
   245: # -- EDITABLE REGION START (lines 246-266) ------------------------------------
   246: def compute_loss(logits, targets, config):
   247:     """Label Smoothing cross-entropy (eps=0.1).
   248: 
   249:     Softens hard targets to (1-eps)*one_hot + eps/C, preventing
   250:     overconfident predictions and improving generalization.
   251:     """
   252:     return F.cross_entropy(logits, targets, label_smoothing=0.1)
   253: # -- EDITABLE REGION END (lines 246-266) --------------------------------------
   254: 
   255: 
```

### `focal_loss` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_loss.py`:

```python
Lines 246–254:
   243: # ============================================================================
   244: 
   245: # -- EDITABLE REGION START (lines 246-266) ------------------------------------
   246: def compute_loss(logits, targets, config):
   247:     """Focal Loss (gamma=2.0).
   248: 
   249:     Modulates CE by (1-pt)^gamma to focus on hard examples,
   250:     reducing the relative loss for well-classified samples.
   251:     """
   252:     ce = F.cross_entropy(logits, targets, reduction='none')
   253:     pt = torch.exp(-ce)
   254:     return ((1 - pt) ** 2.0 * ce).mean()
   255: # -- EDITABLE REGION END (lines 246-266) --------------------------------------
   256: 
   257: 
```

### `poly_loss` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_loss.py`:

```python
Lines 246–254:
   243: # ============================================================================
   244: 
   245: # -- EDITABLE REGION START (lines 246-266) ------------------------------------
   246: def compute_loss(logits, targets, config):
   247:     """PolyLoss (epsilon=2.0).
   248: 
   249:     Adds polynomial correction to CE: CE + eps*(1-pt), where pt is the
   250:     softmax probability assigned to the true class.
   251:     """
   252:     ce = F.cross_entropy(logits, targets)
   253:     pt = F.softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze()
   254:     return ce + 2.0 * (1 - pt).mean()
   255: # -- EDITABLE REGION END (lines 246-266) --------------------------------------
   256: 
   257: 
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
