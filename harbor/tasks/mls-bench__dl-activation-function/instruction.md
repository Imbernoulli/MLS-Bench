# MLS-Bench: dl-activation-function

# DL Activation Function Design

## Research Question
Design an activation function for deep convolutional neural networks that improves generalization performance across different architectures and datasets, while keeping the model definitions, optimizer, initialization, and data pipeline fixed.

## Background
Activation functions introduce nonlinearity into neural networks and critically affect training dynamics, gradient flow, sparsity, and generalization. Classic and modern choices include:

- **ReLU** (Nair & Hinton, 2010): `max(0, x)` — simple, sparse, but zero gradient for negative inputs ("dying ReLU").
- **GELU** (Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)", arXiv:1606.08415): `x * Phi(x)` where `Phi` is the standard Gaussian CDF; smooth weighting by Gaussian probability mass.
- **Swish / SiLU** (Ramachandran, Zoph & Le, "Searching for Activation Functions", arXiv:1710.05941; SiL form due to Elfwing et al., 2017): `x * sigmoid(beta * x)`; self-gated, smooth, non-monotonic. The PyTorch `nn.SiLU` corresponds to `beta = 1`.
- **Mish** (Misra, "Mish: A Self Regularized Non-Monotonic Activation Function", BMVC 2020, arXiv:1908.08681): `x * tanh(softplus(x))`; self-regularized, smooth, non-monotonic.
- **Squared ReLU**, **StarReLU**, and other variants explore polynomial gates and learnable/affine extensions.

These functions differ in smoothness, gating behavior, and negative-domain treatment, and may interact differently with modern network components such as residual connections and batch normalization.

## What You Can Modify
The `CustomActivation` class inside `pytorch-vision/custom_activation.py`. It is an `nn.Module` used as a drop-in replacement for ReLU throughout the network.

You may modify the `forward` computation (any element-wise or channel-wise operation), register learnable parameters in `__init__`, choose any shape of activation curve (monotonic / non-monotonic / bounded), and decide negative-domain behavior (zero, linear, bounded, learnable). Tensor shape must be preserved.

The activation is used in:
- ResNet: BasicBlock (twice per block) and the initial conv.
- VGG: after every Conv-BN pair and inside the classifier head.
- MobileNetV2: replaces the ReLU6 baseline used in inverted residuals.

## Fixed Pipeline
The model definitions, optimizer, schedule, data augmentation, weight initialization, and training loop are fixed by the harness and not editable. Test accuracy is the evaluation metric.

## Baselines
- **gelu** — Hendrycks & Gimpel, arXiv:1606.08415; `nn.GELU` (no learnable parameters).
- **silu** — Ramachandran et al. / Elfwing et al., arXiv:1710.05941; `nn.SiLU`, equivalent to Swish with `beta=1` (no learnable parameters).
- **mish** — Misra, arXiv:1908.08681; `x * tanh(softplus(x))` (no learnable parameters).

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/pytorch-vision/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `pytorch-vision/custom_activation.py`
- editable lines **32–49**




## Readable Context


### `pytorch-vision/custom_activation.py`  [EDITABLE — lines 32–49 only]

```python
     1: """CV Activation Function Benchmark.
     2: 
     3: Train vision models (ResNet, VGG, MobileNetV2) on CIFAR-10/100/FashionMNIST
     4: to evaluate custom activation functions.
     5: 
     6: FIXED: Model architectures, data pipeline, training loop.
     7: EDITABLE: CustomActivation class.
     8: 
     9: Usage:
    10:     python custom_activation.py --arch resnet20 --dataset cifar10 --seed 42
    11:     python custom_activation.py --arch mobilenetv2 --dataset fmnist --seed 42
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
    28: # Custom Activation Function
    29: # ============================================================================
    30: 
    31: # -- EDITABLE REGION START (lines 32-49) --------------------------------------
    32: class CustomActivation(nn.Module):
    33:     """Custom activation function. Drop-in replacement for ReLU.
    34: 
    35:     Input/Output: tensor of any shape, element-wise operation.
    36: 
    37:     Design considerations:
    38:         - Monotonicity vs non-monotonicity tradeoffs
    39:         - Smoothness (differentiability everywhere vs piecewise)
    40:         - Negative-domain behavior (zero, linear, bounded)
    41:         - Computational cost (simple ops vs learned parameters)
    42:         - Interaction with BatchNorm and residual connections
    43:     """
    44: 
    45:     def __init__(self):
    46:         super().__init__()
    47: 
    48:     def forward(self, x):
    49:         return F.relu(x)  # default: ReLU
    50: # -- EDITABLE REGION END (lines 32-49) ----------------------------------------
    51: 
    52: 
    53: # ============================================================================
    54: # Model Architectures (FIXED)
    55: # ============================================================================
    56: 
    57: class BasicBlock(nn.Module):
    58:     """Basic residual block for CIFAR ResNets."""
    59:     expansion = 1
    60: 
    61:     def __init__(self, in_planes, planes, stride=1):
    62:         super().__init__()
    63:         self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
    64:         self.bn1 = nn.BatchNorm2d(planes)
    65:         self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
    66:         self.bn2 = nn.BatchNorm2d(planes)
    67:         self.act = CustomActivation()
    68:         self.shortcut = nn.Sequential()
    69:         if stride != 1 or in_planes != planes * self.expansion:
    70:             self.shortcut = nn.Sequential(
    71:                 nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
    72:                 nn.BatchNorm2d(planes * self.expansion),
    73:             )
    74: 
    75:     def forward(self, x):
    76:         out = self.act(self.bn1(self.conv1(x)))
    77:         out = self.bn2(self.conv2(out))
    78:         out += self.shortcut(x)
    79:         return self.act(out)
    80: 
    81: 
    82: class ResNet(nn.Module):
    83:     """CIFAR-adapted ResNet (He et al., 2016).
    84: 
    85:     Uses 3x3 initial conv (no 7x7), no max pooling, global avg pool at end.
    86:     Standard depths: ResNet-20 ([3,3,3]), ResNet-56 ([9,9,9]).
    87:     """
    88: 
    89:     def __init__(self, block, num_blocks, num_classes=10):
    90:         super().__init__()
    91:         self.in_planes = 16
    92:         self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
    93:         self.bn1 = nn.BatchNorm2d(16)
    94:         self.act = CustomActivation()
    95:         self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
    96:         self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
    97:         self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
    98:         self.fc = nn.Linear(64 * block.expansion, num_classes)
    99: 
   100:     def _make_layer(self, block, planes, num_blocks, stride):
   101:         strides = [stride] + [1] * (num_blocks - 1)
   102:         layers = []
   103:         for s in strides:
   104:             layers.append(block(self.in_planes, planes, s))
   105:             self.in_planes = planes * block.expansion
   106:         return nn.Sequential(*layers)
   107: 
   108:     def forward(self, x):
   109:         out = self.act(self.bn1(self.conv1(x)))
   110:         out = self.layer1(out)
   111:         out = self.layer2(out)
   112:         out = self.layer3(out)
   113:         out = F.adaptive_avg_pool2d(out, 1)
   114:         out = out.view(out.size(0), -1)
   115:         return self.fc(out)
   116: 
   117: 
   118: class VGG(nn.Module):
   119:     """VGG-16 with BatchNorm, adapted for CIFAR (Simonyan & Zisserman, 2015).
   120: 
   121:     Uses adaptive avg pool instead of large FC layers, suitable for 32x32 input.
   122:     """
   123: 
   124:     VGG16_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
   125:                  512, 512, 512, 'M', 512, 512, 512, 'M']
   126: 
   127:     def __init__(self, num_classes=100):
   128:         super().__init__()
   129:         self.features = self._make_layers(self.VGG16_CFG)
   130:         self.classifier = nn.Sequential(
   131:             nn.Linear(512, 512),
   132:             CustomActivation(),
   133:             nn.Dropout(0.5),
   134:             nn.Linear(512, num_classes),
   135:         )
   136: 
   137:     def _make_layers(self, cfg):
   138:         layers = []
   139:         in_channels = 3
   140:         for v in cfg:
   141:             if v == 'M':
   142:                 layers.append(nn.MaxPool2d(2, 2))
   143:             else:
   144:                 layers += [
   145:                     nn.Conv2d(in_channels, v, 3, padding=1),
   146:                     nn.BatchNorm2d(v),
   147:                     CustomActivation(),
   148:                 ]
   149:                 in_channels = v
   150:         return nn.Sequential(*layers)
   151: 
   152:     def forward(self, x):
   153:         x = self.features(x)
   154:         x = F.adaptive_avg_pool2d(x, 1)
   155:         x = x.view(x.size(0), -1)
   156:         return self.classifier(x)
   157: 
   158: 
   159: class InvertedResidual(nn.Module):
   160:     """MobileNetV2 inverted residual block (Sandler et al., 2018)."""
   161: 
   162:     def __init__(self, inp, oup, stride, expand_ratio):
   163:         super().__init__()
   164:         self.stride = stride
   165:         hidden = int(round(inp * expand_ratio))
   166:         self.use_res = (stride == 1 and inp == oup)
   167:         layers = []
   168:         if expand_ratio != 1:
   169:             layers += [
   170:                 nn.Conv2d(inp, hidden, 1, bias=False),
   171:                 nn.BatchNorm2d(hidden),
   172:                 CustomActivation(),
   173:             ]
   174:         layers += [
   175:             nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
   176:             nn.BatchNorm2d(hidden),
   177:             CustomActivation(),
   178:             nn.Conv2d(hidden, oup, 1, bias=False),
   179:             nn.BatchNorm2d(oup),
   180:         ]
   181:         self.conv = nn.Sequential(*layers)
   182: 
   183:     def forward(self, x):
   184:         if self.use_res:
   185:             return x + self.conv(x)
   186:         return self.conv(x)
   187: 
   188: 
   189: class MobileNetV2(nn.Module):
   190:     """MobileNetV2 adapted for CIFAR/small-image input (Sandler et al., 2018).
   191: 
   192:     Uses stride-1 initial conv (no stride-2) for 32x32 input.
   193:     Width multiplier = 1.0, ~2.2M parameters.
   194:     """
   195: 
   196:     CFG = [
   197:         # expand_ratio, channels, num_blocks, stride
   198:         [1, 16, 1, 1],
   199:         [6, 24, 2, 1],
   200:         [6, 32, 3, 2],
   201:         [6, 64, 4, 2],
   202:         [6, 96, 3, 1],
   203:         [6, 160, 3, 2],
   204:         [6, 320, 1, 1],
   205:     ]
   206: 
   207:     def __init__(self, num_classes=10):
   208:         super().__init__()
   209:         self.conv1 = nn.Sequential(
   210:             nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
   211:             nn.BatchNorm2d(32),
   212:             CustomActivation(),
   213:         )
   214:         layers = []
   215:         inp = 32
   216:         for t, c, n, s in self.CFG:
   217:             for i in range(n):
   218:                 stride = s if i == 0 else 1
   219:                 layers.append(InvertedResidual(inp, c, stride, t))
   220:                 inp = c
   221:         self.layers = nn.Sequential(*layers)
   222:         self.conv_last = nn.Sequential(
   223:             nn.Conv2d(320, 1280, 1, bias=False),
   224:             nn.BatchNorm2d(1280),
   225:             CustomActivation(),
   226:         )
   227:         self.fc = nn.Linear(1280, num_classes)
   228: 
   229:     def forward(self, x):
   230:         x = self.conv1(x)
   231:         x = self.layers(x)
   232:         x = self.conv_last(x)
   233:         x = F.adaptive_avg_pool2d(x, 1)
   234:         x = x.view(x.size(0), -1)
   235:         return self.fc(x)
   236: 
   237: 
   238: def build_model(arch, num_classes):
   239:     """Build model by architecture name."""
   240:     if arch == 'resnet20':
   241:         return ResNet(BasicBlock, [3, 3, 3], num_classes)
   242:     elif arch == 'resnet56':
   243:         return ResNet(BasicBlock, [9, 9, 9], num_classes)
   244:     elif arch == 'vgg16bn':
   245:         return VGG(num_classes)
   246:     elif arch == 'mobilenetv2':
   247:         return MobileNetV2(num_classes)
   248:     else:
   249:         raise ValueError(f"Unknown architecture: {arch}")
   250: 
   251: 
   252: # ============================================================================
   253: # Weight Initialization (FIXED)
   254: # ============================================================================
   255: 
   256: def initialize_weights(model):
   257:     """Kaiming initialization for all layers."""
   258:     for m in model.modules():
   259:         if isinstance(m, nn.Conv2d):
   260:             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
   261:         elif isinstance(m, nn.BatchNorm2d):
   262:             nn.init.constant_(m.weight, 1)
   263:             nn.init.constant_(m.bias, 0)
   264:         elif isinstance(m, nn.Linear):
   265:             nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
   266:             if m.bias is not None:
   267:                 nn.init.constant_(m.bias, 0)
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
   331: def train_epoch(model, loader, criterion, optimizer, device):
   332:     """Train for one epoch. Returns (avg_loss, accuracy%)."""
   333:     model.train()
   334:     total_loss, correct, total = 0.0, 0, 0
   335:     for inputs, targets in loader:
   336:         inputs, targets = inputs.to(device), targets.to(device)
   337:         optimizer.zero_grad()
   338:         outputs = model(inputs)
   339:         loss = criterion(outputs, targets)
   340:         loss.backward()
   341:         optimizer.step()
   342:         total_loss += loss.item() * inputs.size(0)
   343:         _, predicted = outputs.max(1)
   344:         correct += predicted.eq(targets).sum().item()
   345:         total += inputs.size(0)
   346:     return total_loss / total, 100.0 * correct / total
   347: 
   348: 
   349: def evaluate(model, loader, criterion, device):
   350:     """Evaluate on test set. Returns (avg_loss, accuracy%)."""
   351:     model.eval()
   352:     total_loss, correct, total = 0.0, 0, 0
   353:     with torch.no_grad():
   354:         for inputs, targets in loader:
   355:             inputs, targets = inputs.to(device), targets.to(device)
   356:             outputs = model(inputs)
   357:             loss = criterion(outputs, targets)
   358:             total_loss += loss.item() * inputs.size(0)
   359:             _, predicted = outputs.max(1)
   360:             correct += predicted.eq(targets).sum().item()
   361:             total += inputs.size(0)
   362:     return total_loss / total, 100.0 * correct / total
   363: 
   364: 
   365: def main():
   366:     parser = argparse.ArgumentParser(description="CV Activation Function Benchmark")
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
   393:     # Initialize
   394:     initialize_weights(model)
   395:     model = model.to(device)
   396: 
   397:     # Optimizer
   398:     criterion = nn.CrossEntropyLoss()
   399:     optimizer = optim.SGD(
   400:         model.parameters(), lr=args.lr,
   401:         momentum=args.momentum, weight_decay=args.weight_decay,
   402:     )
   403:     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
   404: 
   405:     # Train
   406:     best_acc = 0.0
   407:     for epoch in range(args.epochs):
   408:         train_loss, train_acc = train_epoch(
   409:             model, train_loader, criterion, optimizer, device,
   410:         )
   411:         test_loss, test_acc = evaluate(model, test_loader, criterion, device)
   412:         scheduler.step()
   413: 
   414:         if (epoch + 1) % 10 == 0 or epoch == 0:
   415:             print(
   416:                 f"TRAIN_METRICS: epoch={epoch+1} train_loss={train_loss:.4f} "
   417:                 f"train_acc={train_acc:.2f} test_loss={test_loss:.4f} "
   418:                 f"test_acc={test_acc:.2f} lr={optimizer.param_groups[0]['lr']:.6f}",
   419:                 flush=True,
   420:             )
   421: 
   422:         if test_acc > best_acc:
   423:             best_acc = test_acc
   424: 
   425:     print(f"TEST_METRICS: test_acc={best_acc:.2f}", flush=True)
   426: 
   427: 
   428: if __name__ == '__main__':
   429:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `gelu` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_activation.py`:

```python
Lines 32–43:
    29: # ============================================================================
    30: 
    31: # -- EDITABLE REGION START (lines 32-49) --------------------------------------
    32: class CustomActivation(nn.Module):
    33:     """GELU activation function.
    34: 
    35:     GELU(x) = x * Phi(x) where Phi is the Gaussian CDF.
    36:     Smooth, non-monotonic, allows small negative values.
    37:     """
    38: 
    39:     def __init__(self):
    40:         super().__init__()
    41: 
    42:     def forward(self, x):
    43:         return F.gelu(x)
    44: # -- EDITABLE REGION END (lines 32-49) ----------------------------------------
    45: 
    46: 
```

### `silu` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_activation.py`:

```python
Lines 32–43:
    29: # ============================================================================
    30: 
    31: # -- EDITABLE REGION START (lines 32-49) --------------------------------------
    32: class CustomActivation(nn.Module):
    33:     """SiLU/Swish activation function.
    34: 
    35:     SiLU(x) = x * sigmoid(x).
    36:     Self-gated activation discovered via automated search.
    37:     """
    38: 
    39:     def __init__(self):
    40:         super().__init__()
    41: 
    42:     def forward(self, x):
    43:         return F.silu(x)
    44: # -- EDITABLE REGION END (lines 32-49) ----------------------------------------
    45: 
    46: 
```

### `mish` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-vision/custom_activation.py`:

```python
Lines 32–43:
    29: # ============================================================================
    30: 
    31: # -- EDITABLE REGION START (lines 32-49) --------------------------------------
    32: class CustomActivation(nn.Module):
    33:     """Mish activation function.
    34: 
    35:     Mish(x) = x * tanh(softplus(x)).
    36:     Self-regularized non-monotonic activation with smooth gradients.
    37:     """
    38: 
    39:     def __init__(self):
    40:         super().__init__()
    41: 
    42:     def forward(self, x):
    43:         return x * torch.tanh(F.softplus(x))
    44: # -- EDITABLE REGION END (lines 32-49) ----------------------------------------
    45: 
    46: 
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
