# MLS-Bench: cv-data-augmentation

# CV Data Augmentation Strategy Design

## Research Question
Design a training-time data augmentation strategy for image classification that improves test accuracy across different architectures and datasets, while keeping the model architectures, optimizer, test transform, and training loop fixed.

## Background
Data augmentation is a primary regularization tool for training deep networks on limited image data. By applying label-preserving transformations to training images, augmentation increases the effective dataset diversity and shapes the inductive bias of the model. Representative methods include:

- **Standard CIFAR augmentation**: `RandomCrop(32, padding=4)` + `RandomHorizontalFlip` — a minimal geometric baseline.
- **Cutout** (DeVries & Taylor, arXiv:1708.04552): randomly masks square regions of the input, forcing the network to use broader spatial context.
- **RandAugment** (Cubuk et al., CVPR Workshops 2020 / NeurIPS 2020, arXiv:1909.13719): applies `N` randomly selected operations at uniform magnitude `M`, removing the expensive search of AutoAugment-style methods.
- **TrivialAugment** (Müller & Hutter, ICCV 2021, arXiv:2103.10158): applies a single random operation with a random magnitude per image, with no tunable hyperparameters.
- **AugMix** (Hendrycks et al., ICLR 2020, arXiv:1912.02781): mixes multiple augmentation chains for robustness and uncertainty calibration.
- **Random Erasing** (Zhong et al., 2017): an erasing variant closely related to Cutout, often used jointly with other augmentations.

These methods make different choices about geometric, photometric, and masking transforms, and they may behave differently across datasets and model families.

## What You Can Modify
The `build_train_transform(config)` function inside `pytorch-vision/custom_augment.py`. The function receives a `config` dict and must return a `torchvision.transforms.Compose` pipeline.

`config` provides:
- `img_size` (int, `32`)
- `mean` (tuple of channel means)
- `std` (tuple of channel standard deviations)
- `dataset` (str, e.g. `'cifar10'` or `'cifar100'`)

You may use any combination of geometric transforms (crop, flip, rotation, affine, perspective), photometric transforms (color jitter, equalize, posterize, solarize), erasing/masking strategies (cutout, random erasing), automated augmentation policies (AutoAugment, RandAugment, TrivialAugment, AugMix), and custom transform classes defined inside the function. Dataset-specific behavior is allowed.

**Required**: the returned pipeline must include `transforms.ToTensor()` and `transforms.Normalize(config['mean'], config['std'])` so that the produced tensors are normalized as expected by the downstream models. The test-time transform is fixed and is not part of the design space.

## Fixed Pipeline
- Optimizer: SGD with `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`.
- Schedule: cosine annealing over `200` epochs.
- Weight initialization: standard Kaiming normal.
- Evaluation settings: ResNet-20 on CIFAR-10, ResNet-56 on CIFAR-100, MobileNetV2 on FashionMNIST.

## Baselines
- **cutout** — DeVries & Taylor, arXiv:1708.04552; default 16×16 patch on CIFAR-style 32×32 inputs as in the paper.
- **randaugment** — Cubuk et al., arXiv:1909.13719; default `N=2`, `M=14` (paper-reported defaults for ResNet-style models on CIFAR).
- **trivialaugment** — Müller & Hutter, arXiv:2103.10158; parameter-free, single random op per image with random magnitude.

## Metric
Best test accuracy (%, higher is better) achieved during training. The transform must produce normalized tensors compatible with the existing loaders and models, and must not use validation/test labels, change the dataset split, or alter the model and optimization code.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/pytorch-vision/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `pytorch-vision/custom_augment.py`
- editable lines **246–275**




## Readable Context


### `pytorch-vision/custom_augment.py`  [EDITABLE — lines 246–275 only]

```python
     1: """CV Data Augmentation Benchmark.
     2: 
     3: Train vision models (ResNet, VGG, MobileNetV2) on CIFAR-10/100/FashionMNIST to evaluate
     4: data augmentation strategies.
     5: 
     6: FIXED: Model architectures, weight initialization, test transform, data loading, training loop.
     7: EDITABLE: build_train_transform() function.
     8: 
     9: Usage:
    10:     python custom_augment.py --arch resnet20 --dataset cifar10 --seed 42
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
   228:     """Kaiming normal initialization (standard)."""
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
   242: # Data Augmentation
   243: # ============================================================================
   244: 
   245: # -- EDITABLE REGION START (lines 246-275) ------------------------------------
   246: def build_train_transform(config):
   247:     """Build training data transform pipeline.
   248: 
   249:     Called before creating the training dataset. Must return a complete
   250:     transforms.Compose pipeline including ToTensor() and Normalize().
   251: 
   252:     Args:
   253:         config: dict with keys:
   254:             - img_size: int (32 for CIFAR)
   255:             - mean: tuple of floats (per-channel mean)
   256:             - std: tuple of floats (per-channel std)
   257:             - dataset: str ('cifar10' or 'cifar100')
   258: 
   259:     Returns:
   260:         transforms.Compose -- complete training transform pipeline.
   261: 
   262:     Design considerations:
   263:         - Geometric transforms (crop, flip, rotation, affine)
   264:         - Color/photometric transforms (jitter, equalize, posterize)
   265:         - Erasing/masking strategies (cutout, random erasing)
   266:         - Automated augmentation policies (AutoAugment, RandAugment, TrivialAugment)
   267:         - Mixing strategies applied at the tensor level (after ToTensor)
   268:         - Regularization via input perturbation
   269:     """
   270:     return transforms.Compose([
   271:         transforms.RandomCrop(config['img_size'], padding=4),
   272:         transforms.RandomHorizontalFlip(),
   273:         transforms.ToTensor(),
   274:         transforms.Normalize(config['mean'], config['std']),
   275:     ])
   276: # -- EDITABLE REGION END (lines 246-275) --------------------------------------
   277: 
   278: 
   279: # ============================================================================
   280: # Data Loading (FIXED)
   281: # ============================================================================
   282: 
   283: def get_dataloaders(dataset, data_root, batch_size=128, num_workers=4):
   284:     """Create train/test dataloaders.
   285: 
   286:     Train transform is built by build_train_transform() (editable).
   287:     Test transform is fixed (no augmentation).
   288:     """
   289:     if dataset == 'cifar10':
   290:         mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
   291:         num_classes = 10
   292:         Dataset = torchvision.datasets.CIFAR10
   293:     elif dataset == 'cifar100':
   294:         mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
   295:         num_classes = 100
   296:         Dataset = torchvision.datasets.CIFAR100
   297:     elif dataset == 'fmnist':
   298:         mean, std = (0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)
   299:         num_classes = 10
   300:         Dataset = torchvision.datasets.FashionMNIST
   301:     else:
   302:         raise ValueError(f"Unknown dataset: {dataset}")
   303: 
   304:     is_grayscale = (dataset == 'fmnist')
   305:     _repeat3 = transforms.Lambda(lambda x: x.repeat(3, 1, 1))
   306: 
   307:     config = {
   308:         'img_size': 32,
   309:         'mean': mean,
   310:         'std': std,
   311:         'dataset': dataset,
   312:     }
   313:     train_transform = build_train_transform(config)
   314:     # For grayscale datasets, wrap user transform: Resize + user pipeline + channel repeat
   315:     if is_grayscale:
   316:         user_ops = list(train_transform.transforms)
   317:         # Insert Resize at the front (before any spatial augmentation)
   318:         user_ops.insert(0, transforms.Resize(32))
   319:         # Find where ToTensor is and insert channel repeat right after it
   320:         for i, t in enumerate(user_ops):
   321:             if isinstance(t, transforms.ToTensor):
   322:                 user_ops.insert(i + 1, _repeat3)
   323:                 break
   324:         train_transform = transforms.Compose(user_ops)
   325: 
   326:     if is_grayscale:
   327:         test_transform = transforms.Compose([
   328:             transforms.Resize(32),
   329:             transforms.ToTensor(),
   330:             _repeat3,
   331:             transforms.Normalize(mean, std),
   332:         ])
   333:     else:
   334:         test_transform = transforms.Compose([
   335:             transforms.ToTensor(),
   336:             transforms.Normalize(mean, std),
   337:         ])
   338: 
   339:     train_set = Dataset(root=data_root, train=True, download=False, transform=train_transform)
   340:     test_set = Dataset(root=data_root, train=False, download=False, transform=test_transform)
   341: 
   342:     train_loader = torch.utils.data.DataLoader(
   343:         train_set, batch_size=batch_size, shuffle=True,
   344:         num_workers=num_workers, pin_memory=True,
   345:     )
   346:     test_loader = torch.utils.data.DataLoader(
   347:         test_set, batch_size=batch_size, shuffle=False,
   348:         num_workers=num_workers, pin_memory=True,
   349:     )
   350:     return train_loader, test_loader, num_classes
   351: 
   352: 
   353: # ============================================================================
   354: # Training Loop (FIXED)
   355: # ============================================================================
   356: 
   357: def train_epoch(model, loader, criterion, optimizer, device):
   358:     """Train for one epoch. Returns (avg_loss, accuracy%)."""
   359:     model.train()
   360:     total_loss, correct, total = 0.0, 0, 0
   361:     for inputs, targets in loader:
   362:         inputs, targets = inputs.to(device), targets.to(device)
   363:         optimizer.zero_grad()
   364:         outputs = model(inputs)
   365:         loss = criterion(outputs, targets)
   366:         loss.backward()
   367:         optimizer.step()
   368:         total_loss += loss.item() * inputs.size(0)
   369:         _, predicted = outputs.max(1)
   370:         correct += predicted.eq(targets).sum().item()
   371:         total += inputs.size(0)
   372:     return total_loss / total, 100.0 * correct / total
   373: 
   374: 
   375: def evaluate(model, loader, criterion, device):
   376:     """Evaluate on test set. Returns (avg_loss, accuracy%)."""
   377:     model.eval()
   378:     total_loss, correct, total = 0.0, 0, 0
   379:     with torch.no_grad():
   380:         for inputs, targets in loader:
   381:             inputs, targets = inputs.to(device), targets.to(device)
   382:             outputs = model(inputs)
   383:             loss = criterion(outputs, targets)
   384:             total_loss += loss.item() * inputs.size(0)
   385:             _, predicted = outputs.max(1)
   386:             correct += predicted.eq(targets).sum().item()
   387:             total += inputs.size(0)
   388:     return total_loss / total, 100.0 * correct / total
   389: 
   390: 
   391: def main():
   392:     parser = argparse.ArgumentParser(description="CV Data Augmentation Benchmark")
   393:     parser.add_argument('--arch', type=str, required=True,
   394:                         choices=['resnet20', 'resnet56', 'vgg16bn', 'mobilenetv2'])
   395:     parser.add_argument('--dataset', type=str, required=True,
   396:                         choices=['cifar10', 'cifar100', 'fmnist'])
   397:     parser.add_argument('--data-root', type=str, default='/data/cifar')
   398:     parser.add_argument('--epochs', type=int, default=200)
   399:     parser.add_argument('--batch-size', type=int, default=128)
   400:     parser.add_argument('--lr', type=float, default=0.1)
   401:     parser.add_argument('--momentum', type=float, default=0.9)
   402:     parser.add_argument('--weight-decay', type=float, default=5e-4)
   403:     parser.add_argument('--seed', type=int, default=42)
   404:     parser.add_argument('--output-dir', type=str, default='.')
   405:     args = parser.parse_args()
   406: 
   407:     torch.manual_seed(args.seed)
   408:     torch.cuda.manual_seed_all(args.seed)
   409:     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   410: 
   411:     # Data
   412:     train_loader, test_loader, num_classes = get_dataloaders(
   413:         args.dataset, args.data_root, args.batch_size,
   414:     )
   415: 
   416:     # Model
   417:     model = build_model(args.arch, num_classes)
   418: 
   419:     # Initialize
   420:     initialize_weights(model)
   421:     model = model.to(device)
   422: 
   423:     # Optimizer
   424:     criterion = nn.CrossEntropyLoss()
   425:     optimizer = optim.SGD(
   426:         model.parameters(), lr=args.lr,
   427:         momentum=args.momentum, weight_decay=args.weight_decay,
   428:     )
   429:     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
   430: 
   431:     # Train
   432:     best_acc = 0.0
   433:     for epoch in range(args.epochs):
   434:         train_loss, train_acc = train_epoch(
   435:             model, train_loader, criterion, optimizer, device,
   436:         )
   437:         test_loss, test_acc = evaluate(model, test_loader, criterion, device)
   438:         scheduler.step()
   439: 
   440:         if (epoch + 1) % 10 == 0 or epoch == 0:
   441:             print(
   442:                 f"TRAIN_METRICS: epoch={epoch+1} train_loss={train_loss:.4f} "
   443:                 f"train_acc={train_acc:.2f} test_loss={test_loss:.4f} "
   444:                 f"test_acc={test_acc:.2f} lr={optimizer.param_groups[0]['lr']:.6f}",
   445:                 flush=True,
   446:             )
   447: 
   448:         if test_acc > best_acc:
   449:             best_acc = test_acc
   450: 
   451:     print(f"TEST_METRICS: test_acc={best_acc:.2f}", flush=True)
   452: 
   453: 
   454: if __name__ == '__main__':
   455:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `cutout` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_augment.py`:

```python
Lines 246–273:
   243: # ============================================================================
   244: 
   245: # -- EDITABLE REGION START (lines 246-275) ------------------------------------
   246: def build_train_transform(config):
   247:     """Cutout augmentation: random square mask after ToTensor.
   248: 
   249:     Pipeline: RandomCrop + HFlip + ToTensor + Cutout(1, 16) + Normalize.
   250:     """
   251:     class Cutout:
   252:         def __init__(self, n_holes=1, length=16):
   253:             self.n_holes = n_holes
   254:             self.length = length
   255: 
   256:         def __call__(self, img):
   257:             h, w = img.size(1), img.size(2)
   258:             mask = torch.ones_like(img)
   259:             for _ in range(self.n_holes):
   260:                 y = torch.randint(0, h, (1,)).item()
   261:                 x = torch.randint(0, w, (1,)).item()
   262:                 y1, y2 = max(0, y - self.length // 2), min(h, y + self.length // 2)
   263:                 x1, x2 = max(0, x - self.length // 2), min(w, x + self.length // 2)
   264:                 mask[:, y1:y2, x1:x2] = 0
   265:             return img * mask
   266: 
   267:     return transforms.Compose([
   268:         transforms.RandomCrop(config['img_size'], padding=4),
   269:         transforms.RandomHorizontalFlip(),
   270:         transforms.ToTensor(),
   271:         Cutout(n_holes=1, length=16),
   272:         transforms.Normalize(config['mean'], config['std']),
   273:     ])
   274: # -- EDITABLE REGION END (lines 246-275) --------------------------------------
   275: 
   276: 
```

### `randaugment` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_augment.py`:

```python
Lines 246–257:
   243: # ============================================================================
   244: 
   245: # -- EDITABLE REGION START (lines 246-275) ------------------------------------
   246: def build_train_transform(config):
   247:     """RandAugment augmentation: automated policy before geometric transforms.
   248: 
   249:     Pipeline: RandAugment(2, 9) + RandomCrop + HFlip + ToTensor + Normalize.
   250:     """
   251:     return transforms.Compose([
   252:         transforms.RandAugment(num_ops=2, magnitude=9),
   253:         transforms.RandomCrop(config['img_size'], padding=4),
   254:         transforms.RandomHorizontalFlip(),
   255:         transforms.ToTensor(),
   256:         transforms.Normalize(config['mean'], config['std']),
   257:     ])
   258: # -- EDITABLE REGION END (lines 246-275) --------------------------------------
   259: 
   260: 
```

### `trivialaugment` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_augment.py`:

```python
Lines 246–257:
   243: # ============================================================================
   244: 
   245: # -- EDITABLE REGION START (lines 246-275) ------------------------------------
   246: def build_train_transform(config):
   247:     """TrivialAugmentWide: single random op with random magnitude.
   248: 
   249:     Pipeline: TrivialAugmentWide() + RandomCrop + HFlip + ToTensor + Normalize.
   250:     """
   251:     return transforms.Compose([
   252:         transforms.TrivialAugmentWide(),
   253:         transforms.RandomCrop(config['img_size'], padding=4),
   254:         transforms.RandomHorizontalFlip(),
   255:         transforms.ToTensor(),
   256:         transforms.Normalize(config['mean'], config['std']),
   257:     ])
   258: # -- EDITABLE REGION END (lines 246-275) --------------------------------------
   259: 
   260: 
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
