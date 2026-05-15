# MLS-Bench: dl-regularization

# DL Regularization Strategy Design

## Research Question
Design an additional regularization term for deep convolutional image classifiers that improves generalization (test accuracy) across different architectures and datasets, while the main cross-entropy objective, optimizer, and outer training loop remain fixed.

## Background
Beyond standard weight decay (L2 penalty applied through the optimizer), many regularization techniques have been proposed to improve generalization in deep networks:

- **DropBlock** (Ghiasi, Lin & Le, NeurIPS 2018, arXiv:1810.12890): drops contiguous regions of feature maps. Reformulating its core insight as a loss-based penalty yields a spatial co-activation penalty that discourages reliance on contiguous regions without modifying the model graph.
- **Confidence penalty** (Pereyra et al., "Regularizing Neural Networks by Penalizing Confident Output Distributions", arXiv:1701.06548): adds a penalty `−H(p_θ(y|x))` to discourage low-entropy output distributions.
- **Orthogonal regularization** (Brock et al., "Neural Photo Editing with Introspective Adversarial Networks", ICLR 2017, arXiv:1609.07093): encourages weight matrices to be orthogonal via `||W^T W − I||_F^2` (or its soft variants), preserving signal norms.
- **Spectral / Frobenius penalties**: bound the Lipschitz constant or norms of layer weights.

These methods typically apply a fixed penalty throughout training and do not adapt to training dynamics, model architecture, or interactions between different layer types. There is room for regularizers that are more adaptive, architecture-aware, or that combine complementary penalties.

## What You Can Modify
The `compute_regularization(model, inputs, outputs, targets, config)` function inside `pytorch-vision/custom_reg.py`. The function is called every training step and returns a scalar tensor that is added to the cross-entropy loss.

Inputs:
- `model`: the full `nn.Module`. Iterate over `model.named_parameters()` or `model.named_modules()` for weight-based penalties.
- `inputs`: `[B, 3, 32, 32]` input batch (for input-dependent regularization).
- `outputs`: `[B, num_classes]` model logits (for output-based penalties such as confidence/entropy).
- `targets`: `[B]` integer class labels.
- `config`: dict with `num_classes` (int), `epoch` (int, 0-indexed), `total_epochs` (int).

Design directions: weight-based (L1/L2 norms, orthogonality, spectral norms, weight correlation), output-based (entropy, confidence penalty, label-smoothing-style penalties, logit penalties), activation-based (sparsity, diversity via forward hooks), epoch-dependent (warm-up schedules, annealing, curriculum), or architecture-aware (different penalties for conv vs linear, depth-dependent scaling). The returned term must be differentiable.

Note: standard L2 weight decay (`5e-4`) is **already** applied via the optimizer. Your regularization term is *additional*.

## Fixed Pipeline
- Optimizer: SGD with `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`.
- Schedule: cosine annealing over `200` epochs.
- Data augmentation: `RandomCrop(32, pad=4)` + `RandomHorizontalFlip`.
- Evaluation settings: ResNet-56 on CIFAR-100, VGG-16-BN on CIFAR-100, MobileNetV2 on FashionMNIST.

## Baselines
- **dropblock** — Ghiasi et al., arXiv:1810.12890; loss-based DropBlock-inspired co-activation penalty.
- **confidence_penalty** — Pereyra et al., arXiv:1701.06548; default penalty weight `beta=0.1` (within the `[0.1, 1.0]` range explored in the paper).
- **orthogonal_reg** — Brock et al., arXiv:1609.07093; soft orthogonality penalty `||W^T W − I||_F^2` on conv weights with default coefficient `1e-4`.

## Metric
Best test accuracy (%, higher is better) achieved during training. The regularizer must remain differentiable, computationally reasonable, and must not alter the dataset, architecture, base loss, optimizer, scheduler, or evaluation procedure.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/pytorch-vision/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `pytorch-vision/custom_reg.py`
- editable lines **246–273**




## Readable Context


### `pytorch-vision/custom_reg.py`  [EDITABLE — lines 246–273 only]

```python
     1: """CV Regularization Benchmark.
     2: 
     3: Train vision models (ResNet, VGG, MobileNetV2) on CIFAR-10/100/FashionMNIST to evaluate
     4: regularization strategies.
     5: 
     6: FIXED: Model architectures, weight initialization, data pipeline, training loop.
     7: EDITABLE: compute_regularization() function.
     8: 
     9: Usage:
    10:     python custom_reg.py --arch resnet20 --dataset cifar10 --seed 42
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
   228:     """Kaiming initialization for all layers."""
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
   242: # Regularization
   243: # ============================================================================
   244: 
   245: # -- EDITABLE REGION START (lines 246-273) ------------------------------------
   246: def compute_regularization(model, inputs, outputs, targets, config):
   247:     """Compute regularization loss term added to classification loss.
   248: 
   249:     Called every training step. The returned scalar is added to the
   250:     cross-entropy loss before backpropagation.
   251: 
   252:     Args:
   253:         model: the network (nn.Module, ResNet or VGG)
   254:         inputs: [B, 3, 32, 32] input batch
   255:         outputs: [B, num_classes] model logits
   256:         targets: [B] integer class labels
   257:         config: dict with keys:
   258:             - 'num_classes': int (10 or 100)
   259:             - 'epoch': int (current epoch, 0-indexed)
   260:             - 'total_epochs': int (total training epochs)
   261: 
   262:     Returns:
   263:         scalar torch.Tensor -- regularization loss (added to CE loss).
   264:         Return 0 for no regularization.
   265: 
   266:     Design considerations:
   267:         - Weight-based penalties (L1, L2, orthogonality, spectral)
   268:         - Output-based penalties (confidence, entropy, label smoothing)
   269:         - Activation-based penalties (sparsity, diversity)
   270:         - Gradient-based penalties (gradient penalty, Jacobian)
   271:         - Epoch-dependent scheduling (warm-up, annealing)
   272:     """
   273:     raise NotImplementedError("Implement your regularization strategy")
   274: # -- EDITABLE REGION END (lines 246-273) --------------------------------------
   275: 
   276: 
   277: # ============================================================================
   278: # Data Loading (FIXED)
   279: # ============================================================================
   280: 
   281: def get_dataloaders(dataset, data_root, batch_size=128, num_workers=4):
   282:     """Create train/test dataloaders with standard augmentation."""
   283:     if dataset == 'cifar10':
   284:         mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
   285:         num_classes = 10
   286:         Dataset = torchvision.datasets.CIFAR10
   287:     elif dataset == 'cifar100':
   288:         mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
   289:         num_classes = 100
   290:         Dataset = torchvision.datasets.CIFAR100
   291:     elif dataset == 'fmnist':
   292:         mean, std = (0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)
   293:         num_classes = 10
   294:         Dataset = torchvision.datasets.FashionMNIST
   295:     else:
   296:         raise ValueError(f"Unknown dataset: {dataset}")
   297: 
   298:     is_grayscale = (dataset == 'fmnist')
   299: 
   300:     train_transform_list = [
   301:         transforms.Resize(32),
   302:         transforms.RandomCrop(32, padding=4),
   303:         transforms.RandomHorizontalFlip(),
   304:         transforms.ToTensor(),
   305:     ]
   306:     if is_grayscale:
   307:         train_transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
   308:     train_transform_list.append(transforms.Normalize(mean, std))
   309:     train_transform = transforms.Compose(train_transform_list)
   310: 
   311:     test_transform_list = [
   312:         transforms.Resize(32),
   313:         transforms.ToTensor(),
   314:     ]
   315:     if is_grayscale:
   316:         test_transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
   317:     test_transform_list.append(transforms.Normalize(mean, std))
   318:     test_transform = transforms.Compose(test_transform_list)
   319: 
   320:     train_set = Dataset(root=data_root, train=True, download=False, transform=train_transform)
   321:     test_set = Dataset(root=data_root, train=False, download=False, transform=test_transform)
   322: 
   323:     train_loader = torch.utils.data.DataLoader(
   324:         train_set, batch_size=batch_size, shuffle=True,
   325:         num_workers=num_workers, pin_memory=True,
   326:     )
   327:     test_loader = torch.utils.data.DataLoader(
   328:         test_set, batch_size=batch_size, shuffle=False,
   329:         num_workers=num_workers, pin_memory=True,
   330:     )
   331:     return train_loader, test_loader, num_classes
   332: 
   333: 
   334: # ============================================================================
   335: # Training Loop (FIXED)
   336: # ============================================================================
   337: 
   338: def train_epoch(model, loader, criterion, optimizer, device, config):
   339:     """Train for one epoch with regularization. Returns (avg_loss, accuracy%)."""
   340:     model.train()
   341:     total_loss, correct, total = 0.0, 0, 0
   342:     for inputs, targets in loader:
   343:         inputs, targets = inputs.to(device), targets.to(device)
   344:         optimizer.zero_grad()
   345:         outputs = model(inputs)
   346:         loss = criterion(outputs, targets) + compute_regularization(
   347:             model, inputs, outputs, targets, config,
   348:         )
   349:         loss.backward()
   350:         optimizer.step()
   351:         total_loss += loss.item() * inputs.size(0)
   352:         _, predicted = outputs.max(1)
   353:         correct += predicted.eq(targets).sum().item()
   354:         total += inputs.size(0)
   355:     return total_loss / total, 100.0 * correct / total
   356: 
   357: 
   358: def evaluate(model, loader, criterion, device):
   359:     """Evaluate on test set. Returns (avg_loss, accuracy%)."""
   360:     model.eval()
   361:     total_loss, correct, total = 0.0, 0, 0
   362:     with torch.no_grad():
   363:         for inputs, targets in loader:
   364:             inputs, targets = inputs.to(device), targets.to(device)
   365:             outputs = model(inputs)
   366:             loss = criterion(outputs, targets)
   367:             total_loss += loss.item() * inputs.size(0)
   368:             _, predicted = outputs.max(1)
   369:             correct += predicted.eq(targets).sum().item()
   370:             total += inputs.size(0)
   371:     return total_loss / total, 100.0 * correct / total
   372: 
   373: 
   374: def main():
   375:     parser = argparse.ArgumentParser(description="CV Regularization Benchmark")
   376:     parser.add_argument('--arch', type=str, required=True,
   377:                         choices=['resnet20', 'resnet56', 'vgg16bn', 'mobilenetv2'])
   378:     parser.add_argument('--dataset', type=str, required=True,
   379:                         choices=['cifar10', 'cifar100', 'fmnist'])
   380:     parser.add_argument('--data-root', type=str, default='/data/cifar')
   381:     parser.add_argument('--epochs', type=int, default=200)
   382:     parser.add_argument('--batch-size', type=int, default=128)
   383:     parser.add_argument('--lr', type=float, default=0.1)
   384:     parser.add_argument('--momentum', type=float, default=0.9)
   385:     parser.add_argument('--weight-decay', type=float, default=0)
   386:     parser.add_argument('--seed', type=int, default=42)
   387:     parser.add_argument('--output-dir', type=str, default='.')
   388:     args = parser.parse_args()
   389: 
   390:     torch.manual_seed(args.seed)
   391:     torch.cuda.manual_seed_all(args.seed)
   392:     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   393: 
   394:     # Data
   395:     train_loader, test_loader, num_classes = get_dataloaders(
   396:         args.dataset, args.data_root, args.batch_size,
   397:     )
   398: 
   399:     # Model
   400:     model = build_model(args.arch, num_classes)
   401:     initialize_weights(model)
   402:     model = model.to(device)
   403: 
   404:     # Optimizer
   405:     criterion = nn.CrossEntropyLoss()
   406:     optimizer = optim.SGD(
   407:         model.parameters(), lr=args.lr,
   408:         momentum=args.momentum, weight_decay=args.weight_decay,
   409:     )
   410:     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
   411: 
   412:     # Config for regularization
   413:     config = {
   414:         'num_classes': num_classes,
   415:         'epoch': 0,
   416:         'total_epochs': args.epochs,
   417:     }
   418: 
   419:     # Train
   420:     best_acc = 0.0
   421:     for epoch in range(args.epochs):
   422:         config['epoch'] = epoch
   423:         train_loss, train_acc = train_epoch(
   424:             model, train_loader, criterion, optimizer, device, config,
   425:         )
   426:         test_loss, test_acc = evaluate(model, test_loader, criterion, device)
   427:         scheduler.step()
   428: 
   429:         if (epoch + 1) % 10 == 0 or epoch == 0:
   430:             print(
   431:                 f"TRAIN_METRICS: epoch={epoch+1} train_loss={train_loss:.4f} "
   432:                 f"train_acc={train_acc:.2f} test_loss={test_loss:.4f} "
   433:                 f"test_acc={test_acc:.2f} lr={optimizer.param_groups[0]['lr']:.6f}",
   434:                 flush=True,
   435:             )
   436: 
   437:         if test_acc > best_acc:
   438:             best_acc = test_acc
   439: 
   440:     print(f"TEST_METRICS: test_acc={best_acc:.2f}", flush=True)
   441: 
   442: 
   443: if __name__ == '__main__':
   444:     main()
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **resnet56-cifar100** — wall-clock budget `02:00:00`, compute share `1.0`
- **vgg16bn-cifar100** — wall-clock budget `02:00:00`, compute share `1.0`
- **mobilenetv2-fmnist** — wall-clock budget `02:00:00`, compute share `1.0`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `dropblock` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_reg.py`:

```python
Lines 246–287:
   243: # ============================================================================
   244: 
   245: # -- EDITABLE REGION START (lines 246-273) ------------------------------------
   246: def compute_regularization(model, inputs, outputs, targets, config):
   247:     """Spatial co-activation penalty on convolutional weights.
   248: 
   249:     Applies a spatial co-activation penalty on convolutional weights.
   250:     For each Conv2d layer with spatial kernels >= block_size, it
   251:     penalizes the mean energy of local spatial blocks in the weight
   252:     tensor, discouraging spatially correlated filter patterns.
   253: 
   254:     Uses conservative strength (lambda_max=1e-4) with linear warm-up
   255:     and only activates after 20% of training to avoid destabilizing
   256:     early learning, particularly for BatchNorm-heavy architectures.
   257: 
   258:     block_size=3, lambda_max=1e-4, linear warm-up with delayed start.
   259:     """
   260:     block_size = 3
   261:     lambda_max = 1e-4
   262:     progress = config['epoch'] / max(config['total_epochs'] - 1, 1)
   263: 
   264:     # Delay activation: no penalty for first 20% of training
   265:     if progress < 0.2:
   266:         return torch.tensor(0.0, device=outputs.device)
   267: 
   268:     # Linear schedule from 20% to 100% of training
   269:     adjusted_progress = (progress - 0.2) / 0.8
   270:     lam = lambda_max * adjusted_progress
   271: 
   272:     reg = torch.tensor(0.0, device=outputs.device)
   273:     count = 0
   274:     for m in model.modules():
   275:         if isinstance(m, nn.Conv2d) and m.kernel_size[0] >= block_size:
   276:             w = m.weight  # [out_c, in_c, kH, kW]
   277:             if w.size(-1) >= block_size and w.size(-2) >= block_size:
   278:                 # Mean squared magnitude within spatial blocks
   279:                 w_sq = w.pow(2).mean(dim=1, keepdim=True)  # [out_c, 1, kH, kW]
   280:                 pad = block_size // 2
   281:                 local = F.avg_pool2d(w_sq, block_size, stride=1, padding=pad)
   282:                 reg = reg + local.mean()
   283:                 count += 1
   284: 
   285:     if count > 0:
   286:         reg = reg / count
   287:     return lam * reg
   288: # -- EDITABLE REGION END (lines 246-273) --------------------------------------
   289: 
   290: 
```

### `confidence_penalty` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_reg.py`:

```python
Lines 246–255:
   243: # ============================================================================
   244: 
   245: # -- EDITABLE REGION START (lines 246-273) ------------------------------------
   246: def compute_regularization(model, inputs, outputs, targets, config):
   247:     """Confidence penalty: penalize low-entropy predictions.
   248: 
   249:     Computes negative entropy of the softmax distribution and adds it
   250:     as a penalty, encouraging the model to be less over-confident.
   251:     Beta=0.1.
   252:     """
   253:     probs = F.softmax(outputs, dim=-1)
   254:     entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
   255:     return -0.1 * entropy  # penalize confident (low-entropy) predictions
   256: # -- EDITABLE REGION END (lines 246-273) --------------------------------------
   257: 
   258: 
```

### `orthogonal_reg` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_reg.py`:

```python
Lines 246–259:
   243: # ============================================================================
   244: 
   245: # -- EDITABLE REGION START (lines 246-273) ------------------------------------
   246: def compute_regularization(model, inputs, outputs, targets, config):
   247:     """Orthogonal regularization on convolutional weights.
   248: 
   249:     Penalizes deviation from orthogonality: ||W^T W - I||_F^2 for each
   250:     4D conv weight reshaped to [out_channels, in*k*k]. Coefficient=1e-4.
   251:     """
   252:     reg = torch.tensor(0.0, device=outputs.device)
   253:     for name, p in model.named_parameters():
   254:         if 'conv' in name and 'weight' in name and p.dim() == 4:
   255:             W = p.view(p.size(0), -1)  # [out, in*k*k]
   256:             WtW = W @ W.t()
   257:             I = torch.eye(W.size(0), device=W.device)
   258:             reg = reg + ((WtW - I) ** 2).sum()
   259:     return 1e-4 * reg
   260: # -- EDITABLE REGION END (lines 246-273) --------------------------------------
   261: 
   262: 
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
