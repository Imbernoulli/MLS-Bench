# MLS-Bench: jepa-regularizer

# JEPA Self-Supervised Learning: Anti-Collapse Regularization

## Research Question
Design an improved anti-collapse regularization loss for Joint Embedding Predictive Architecture (JEPA) self-supervised image representation learning. Your regularizer should prevent representation collapse (where all inputs map to the same output) while encouraging the model to learn useful, discriminative features.

## Background
JEPA / joint-embedding self-supervised methods (Assran et al., I-JEPA, CVPR 2023, arXiv:2301.08243) optimize an invariance objective that, on its own, admits the trivial solution where the encoder maps every input to a constant. Anti-collapse regularizers solve this in different ways:
- **VICReg** (Bardes, Ponce, LeCun, ICLR 2022, arXiv:2105.04906) combines a per-dimension variance hinge, a covariance off-diagonal penalty, and an MSE invariance term.
- **Barlow Twins** decorrelates the cross-correlation matrix between two views.
- **Whitening / decorrelation** approaches enforce identity covariance directly.

The choice of regularizer determines what representation geometry is preferred and how it transfers to downstream linear probing.

## What You Can Modify
The editable region in `custom_regularizer.py` is the `CustomRegularizer` class plus the `CONFIG_OVERRIDES` dictionary. The class receives two projected embedding tensors from different augmented views of the same images and must return a loss dictionary.

Interface:
- **Input**: `z1: [B, D]` and `z2: [B, D]` — projected embeddings from two augmented views
- **Output**: `dict` with at least a `"loss"` key containing a scalar tensor

You may add any parameters to `__init__`, define helper methods, and use any PyTorch operations. The imports at the top of the file (torch, torch.nn, torch.nn.functional, etc.) are available.

## Evaluation
- **Metric**: `val_acc` — linear probe classification accuracy on CIFAR-10 (higher is better)
- **Benchmarks**: three backbone architectures (ResNet-18, ResNet-34, ResNet-50) test regularizer generalization across model scales
- **Projector**: features_dim → 2048 → 2048 MLP
- **Training**: 100 epochs, batch size 256, LARS optimizer (lr=0.3), warmup cosine schedule
- **Dataset**: CIFAR-10 (50k train / 10k val)


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/eb_jepa/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `eb_jepa/custom_regularizer.py`
- editable lines **33–58**


Other files you may **read** for context (do not modify):
- `eb_jepa/losses.py`
- `eb_jepa/jepa.py`


## Readable Context


### `eb_jepa/custom_regularizer.py`  [EDITABLE — lines 33–58 only]

```python
     1: """
     2: CIFAR-10 JEPA Self-Supervised Training Script (Self-Contained)
     3: 
     4: Trains a ResNet-18 backbone with a projector using a two-view augmentation
     5: pipeline and an anti-collapse regularization loss. Evaluation is performed
     6: via an online linear probe on CIFAR-10 validation set.
     7: 
     8: Usage:
     9:     python custom_regularizer.py
    10: """
    11: 
    12: import sys; sys.path = [p for p in sys.path if not __import__('os').path.isfile(__import__('os').path.join(p, 'logging.py'))]
    13: import os
    14: import math
    15: import time
    16: import random
    17: 
    18: import numpy as np
    19: import torch
    20: import torch.nn as nn
    21: import torch.nn.functional as F
    22: import torch.optim as optim
    23: import torchvision
    24: import torchvision.transforms as transforms
    25: from torch.amp import GradScaler, autocast
    26: from torch.optim.optimizer import required
    27: from torch.utils.data import DataLoader, Dataset
    28: from torchvision.datasets import CIFAR10
    29: 
    30: 
    31: # ── Custom Regularizer ─────────────────────────────────────────────────────
    32: # EDITABLE REGION START
    33: class CustomRegularizer(nn.Module):
    34:     """Anti-collapse regularizer for self-supervised JEPA learning.
    35: 
    36:     Takes two projected embedding tensors from different augmented views
    37:     and returns a loss dict that prevents representation collapse while
    38:     encouraging useful feature learning.
    39: 
    40:     Args:
    41:         z1: [B, D] projected embeddings from view 1
    42:         z2: [B, D] projected embeddings from view 2
    43: 
    44:     Returns:
    45:         dict with at least a "loss" key (scalar tensor)
    46:     """
    47: 
    48:     def __init__(self):
    49:         super().__init__()
    50: 
    51:     def forward(self, z1, z2):
    52:         loss = torch.tensor(0.0, device=z1.device, requires_grad=True)
    53:         return {"loss": loss}
    54: 
    55: 
    56: # CONFIG_OVERRIDES: override training hyperparameters for your method.
    57: # Allowed keys: proj_output_dim, proj_hidden_dim.
    58: CONFIG_OVERRIDES = {}
    59: # EDITABLE REGION END
    60: 
    61: 
    62: # ── Backbone ──────────────────────────────────────────────────────────────
    63: 
    64: def build_backbone(arch="resnet18"):
    65:     """Build a backbone modified for CIFAR-10 (small 3x3 conv1, no maxpool).
    66: 
    67:     Returns (backbone_module, features_dim).
    68:     """
    69:     builder = {
    70:         "resnet18": (torchvision.models.resnet18, 512),
    71:         "resnet34": (torchvision.models.resnet34, 512),
    72:         "resnet50": (torchvision.models.resnet50, 2048),
    73:     }
    74:     if arch not in builder:
    75:         raise ValueError(f"Unknown ARCH={arch!r}. Choose from {list(builder)}")
    76:     fn, features_dim = builder[arch]
    77:     model = fn()
    78:     model.fc = nn.Identity()
    79:     model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    80:     model.maxpool = nn.Identity()
    81:     return model, features_dim
    82: 
    83: 
    84: # ── ImageSSL Model ──────────────────────────────────────────────────────────
    85: 
    86: class ImageSSL(nn.Module):
    87:     """Image Self-Supervised Learning model with backbone + projector."""
    88: 
    89:     def __init__(
    90:         self, backbone, features_dim, proj_hidden_dim=2048, proj_output_dim=2048
    91:     ):
    92:         super().__init__()
    93:         self.backbone = backbone
    94:         self.features_dim = features_dim
    95:         self.projector = nn.Sequential(
    96:             nn.Linear(features_dim, proj_hidden_dim),
    97:             nn.BatchNorm1d(proj_hidden_dim),
    98:             nn.ReLU(),
    99:             nn.Linear(proj_hidden_dim, proj_hidden_dim),
   100:             nn.BatchNorm1d(proj_hidden_dim),
   101:             nn.ReLU(),
   102:             nn.Linear(proj_hidden_dim, proj_output_dim),
   103:         )
   104: 
   105:     def forward(self, x):
   106:         features = self.backbone(x)
   107:         projections = self.projector(features)
   108:         return features, projections
   109: 
   110: 
   111: # ── Linear Probe ────────────────────────────────────────────────────────────
   112: 
   113: class LinearProbe(nn.Module):
   114:     """Linear probe classifier for evaluating representations."""
   115: 
   116:     def __init__(self, feature_dim, num_classes):
   117:         super().__init__()
   118:         self.classifier = nn.Linear(feature_dim, num_classes)
   119: 
   120:     def forward(self, x):
   121:         return self.classifier(x)
   122: 
   123: 
   124: # ── LARS Optimizer ──────────────────────────────────────────────────────────
   125: 
   126: class LARS(optim.Optimizer):
   127:     """LARS (Layer-wise Adaptive Rate Scaling) optimizer."""
   128: 
   129:     def __init__(
   130:         self,
   131:         params,
   132:         lr=required,
   133:         momentum=0,
   134:         dampening=0,
   135:         weight_decay=0,
   136:         nesterov=False,
   137:         eta=1e-3,
   138:         eps=1e-8,
   139:         clip_lr=False,
   140:         exclude_bias_n_norm=False,
   141:     ):
   142:         if lr is not required and lr < 0.0:
   143:             raise ValueError(f"Invalid learning rate: {lr}")
   144:         if momentum < 0.0:
   145:             raise ValueError(f"Invalid momentum value: {momentum}")
   146:         if weight_decay < 0.0:
   147:             raise ValueError(f"Invalid weight_decay value: {weight_decay}")
   148: 
   149:         defaults = dict(
   150:             lr=lr,
   151:             momentum=momentum,
   152:             dampening=dampening,
   153:             weight_decay=weight_decay,
   154:             nesterov=nesterov,
   155:             eta=eta,
   156:             eps=eps,
   157:             clip_lr=clip_lr,
   158:             exclude_bias_n_norm=exclude_bias_n_norm,
   159:         )
   160:         if nesterov and (momentum <= 0 or dampening != 0):
   161:             raise ValueError("Nesterov momentum requires a momentum and zero dampening")
   162:         super().__init__(params, defaults)
   163: 
   164:     def __setstate__(self, state):
   165:         super().__setstate__(state)
   166:         for group in self.param_groups:
   167:             group.setdefault("nesterov", False)
   168: 
   169:     @torch.no_grad()
   170:     def step(self, closure=None):
   171:         loss = None
   172:         if closure is not None:
   173:             with torch.enable_grad():
   174:                 loss = closure()
   175: 
   176:         for group in self.param_groups:
   177:             weight_decay = group["weight_decay"]
   178:             momentum = group["momentum"]
   179:             dampening = group["dampening"]
   180:             nesterov = group["nesterov"]
   181: 
   182:             for p in group["params"]:
   183:                 if p.grad is None:
   184:                     continue
   185: 
   186:                 d_p = p.grad
   187:                 p_norm = torch.norm(p.data)
   188:                 g_norm = torch.norm(p.grad.data)
   189: 
   190:                 if p.ndim != 1 or not group["exclude_bias_n_norm"]:
   191:                     if p_norm != 0 and g_norm != 0:
   192:                         lars_lr = p_norm / (
   193:                             g_norm + p_norm * weight_decay + group["eps"]
   194:                         )
   195:                         lars_lr *= group["eta"]
   196: 
   197:                         if group["clip_lr"]:
   198:                             lars_lr = min(lars_lr / group["lr"], 1)
   199: 
   200:                         d_p = d_p.add(p, alpha=weight_decay)
   201:                         d_p *= lars_lr
   202: 
   203:                 if momentum != 0:
   204:                     param_state = self.state[p]
   205:                     if "momentum_buffer" not in param_state:
   206:                         buf = param_state["momentum_buffer"] = torch.clone(
   207:                             d_p
   208:                         ).detach()
   209:                     else:
   210:                         buf = param_state["momentum_buffer"]
   211:                         buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
   212:                     if nesterov:
   213:                         d_p = d_p.add(buf, alpha=momentum)
   214:                     else:
   215:                         d_p = buf
   216: 
   217:                 p.add_(d_p, alpha=-group["lr"])
   218: 
   219:         return loss
   220: 
   221: 
   222: # ── Warmup Cosine Scheduler ────────────────────────────────────────────────
   223: 
   224: class WarmupCosineScheduler:
   225:     """Warmup cosine learning rate scheduler."""
   226: 
   227:     def __init__(
   228:         self,
   229:         optimizer,
   230:         warmup_epochs,
   231:         max_epochs,
   232:         base_lr,
   233:         min_lr=0.0,
   234:         warmup_start_lr=3e-5,
   235:     ):
   236:         self.optimizer = optimizer
   237:         self.warmup_epochs = warmup_epochs
   238:         self.max_epochs = max_epochs
   239:         self.base_lr = base_lr
   240:         self.min_lr = min_lr
   241:         self.warmup_start_lr = warmup_start_lr
   242: 
   243:     def step(self, epoch):
   244:         if epoch < self.warmup_epochs:
   245:             lr = self.warmup_start_lr + epoch * (
   246:                 self.base_lr - self.warmup_start_lr
   247:             ) / max(self.warmup_epochs - 1, 1)
   248:         else:
   249:             lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
   250:                 1
   251:                 + math.cos(
   252:                     (epoch - self.warmup_epochs)
   253:                     / max(self.max_epochs - self.warmup_epochs, 1)
   254:                     * math.pi
   255:                 )
   256:             )
   257: 
   258:         for param_group in self.optimizer.param_groups:
   259:             param_group["lr"] = lr
   260: 
   261: 
   262: # ── Data Augmentations ──────────────────────────────────────────────────────
   263: 
   264: def get_train_transforms():
   265:     """Get training transforms for self-supervised learning on CIFAR-10."""
   266:     transform = transforms.Compose(
   267:         [
   268:             transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
   269:             transforms.RandomApply(
   270:                 [
   271:                     transforms.ColorJitter(
   272:                         brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
   273:                     )
   274:                 ],
   275:                 p=0.8,
   276:             ),
   277:             transforms.RandomGrayscale(p=0.2),
   278:             # Solarization at p=0.1 — paper's upstream pipeline uses this and
   279:             # its absence disproportionately hurts SIGReg (Gaussianity-on-
   280:             # random-projections benefits from augmentation diversity). See
   281:             # eb_jepa/examples/image_jepa/dataset.py:46-79.
   282:             transforms.RandomSolarize(threshold=128, p=0.1),
   283:             transforms.RandomHorizontalFlip(),
   284:             transforms.ToTensor(),
   285:             transforms.Normalize(
   286:                 (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
   287:             ),
   288:         ]
   289:     )
   290:     return transform
   291: 
   292: 
   293: def get_val_transforms():
   294:     """Get validation transforms."""
   295:     return transforms.Compose(
   296:         [
   297:             transforms.ToTensor(),
   298:             transforms.Normalize(
   299:                 (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
   300:             ),
   301:         ]
   302:     )
   303: 
   304: 
   305: class ImageDataset(Dataset):
   306:     """Dataset that applies augmentations multiple times to create views."""
   307: 
   308:     def __init__(self, dataset, transform, num_crops=2):
   309:         self.dataset = dataset
   310:         self.transform = transform
   311:         self.num_crops = num_crops
   312: 
   313:     def __len__(self):
   314:         return len(self.dataset)
   315: 
   316:     def __getitem__(self, idx):
   317:         image, label = self.dataset[idx]
   318:         views = [self.transform(image) for _ in range(self.num_crops)]
   319:         return views, label
   320: 
   321: 
   322: # ── Evaluation ──────────────────────────────────────────────────────────────
   323: 
   324: def evaluate_linear_probe(model, linear_probe, val_loader, device, use_amp=True):
   325:     """Evaluate linear probe on validation set."""
   326:     model.eval()
   327:     linear_probe.eval()
   328: 
   329:     total_loss = 0
   330:     correct = 0
   331:     total = 0
   332: 
   333:     with torch.no_grad():
   334:         for data, target in val_loader:
   335:             data = data.to(device, non_blocking=True)
   336:             target = target.to(device, non_blocking=True)
   337: 
   338:             with autocast("cuda", enabled=use_amp):
   339:                 features, _ = model(data)
   340: 
   341:             outputs = linear_probe(features.float())
   342:             loss = F.cross_entropy(outputs, target)
   343: 
   344:             total_loss += loss.item()
   345:             _, predicted = outputs.max(1)
   346:             total += target.size(0)
   347:             correct += predicted.eq(target).sum().item()
   348: 
   349:     accuracy = 100.0 * correct / total
   350:     avg_loss = total_loss / len(val_loader)
   351:     return accuracy, avg_loss
   352: 
   353: 
   354: # ── Training Loop ───────────────────────────────────────────────────────────
   355: 
   356: def train_epoch(
   357:     model,
   358:     train_loader,
   359:     optimizer,
   360:     scheduler,
   361:     linear_probe,
   362:     scaler,
   363:     device,
   364:     epoch,
   365:     loss_fn,
   366:     use_amp=True,
   367:     dtype=torch.bfloat16,
   368: ):
   369:     """Train for one epoch."""
   370:     model.train()
   371:     linear_probe.train()
   372: 
   373:     loss_totals = {}
   374:     total_linear_loss = 0
   375:     linear_correct = 0
   376:     linear_total = 0
   377: 
   378:     for batch_idx, (views, target) in enumerate(train_loader):
   379:         view1, view2 = views[0].to(device, non_blocking=True), views[1].to(
   380:             device, non_blocking=True
   381:         )
   382:         target = target.to(device, non_blocking=True)
   383: 
   384:         with autocast(device.type, enabled=use_amp, dtype=dtype):
   385:             features, z1 = model(view1)
   386:             _, z2 = model(view2)
   387:             loss_dict = loss_fn(z1, z2)
   388:             loss = loss_dict["loss"]
   389: 
   390:         with torch.no_grad():
   391:             features_frozen = features.detach().float()
   392: 
   393:         linear_outputs = linear_probe(features_frozen)
   394:         linear_loss = F.cross_entropy(linear_outputs, target)
   395: 
   396:         _, predicted = linear_outputs.max(1)
   397:         linear_correct_batch = predicted.eq(target).sum().item()
   398: 
   399:         total_loss_batch = loss + linear_loss
   400: 
   401:         optimizer.zero_grad()
   402:         scaler.scale(total_loss_batch).backward()
   403:         scaler.step(optimizer)
   404:         scaler.update()
   405: 
   406:         for key, value in loss_dict.items():
   407:             if key not in loss_totals:
   408:                 loss_totals[key] = 0
   409:             loss_totals[key] += value.item() if torch.is_tensor(value) else value
   410:         total_linear_loss += linear_loss.item()
   411: 
   412:         linear_total += target.size(0)
   413:         linear_correct += linear_correct_batch
   414: 
   415:     scheduler.step(epoch)
   416: 
   417:     num_batches = len(train_loader)
   418:     metrics = {key: total / num_batches for key, total in loss_totals.items()}
   419:     metrics["linear_loss"] = total_linear_loss / num_batches
   420:     metrics["linear_acc"] = 100.0 * linear_correct / linear_total
   421: 
   422:     return metrics
   423: 
   424: 
   425: # ── Main ────────────────────────────────────────────────────────────────────
   426: 
   427: def seed_everything(seed):
   428:     os.environ["PYTHONHASHSEED"] = str(seed)
   429:     random.seed(seed)
   430:     np.random.seed(seed)
   431:     torch.manual_seed(seed)
   432:     if torch.cuda.is_available():
   433:         torch.cuda.manual_seed(seed)
   434:         torch.cuda.manual_seed_all(seed)
   435:     torch.backends.cudnn.benchmark = False
   436: 
   437: 
   438: def seed_worker(worker_id):
   439:     worker_seed = torch.initial_seed() % 2**32
   440:     random.seed(worker_seed)
   441:     np.random.seed(worker_seed)
   442: 
   443: 
   444: def make_generator(seed):
   445:     generator = torch.Generator()
   446:     generator.manual_seed(seed)
   447:     return generator
   448: 
   449: 
   450: def main():
   451:     # Configuration
   452:     seed = int(os.environ.get("SEED", 42))
   453:     arch = os.environ.get("ARCH", "resnet18")
   454:     data_dir = os.environ.get("EBJEPA_DSETS", "/data/eb_jepa")
   455:     # Match the upstream image_jepa SIGReg/VICReg comparisons, which train
   456:     # for 300 epochs on CIFAR-10. At 100 epochs both baselines under-converge
   457:     # and the expected SIGReg/VICReg ordering is not reliable.
   458:     epochs = 300
   459:     batch_size = 256
   460:     lr = 0.3
   461:     weight_decay = 1e-4
   462:     warmup_epochs = 10
   463:     min_lr = 0.0
   464:     warmup_start_lr = 3e-5
   465:     num_workers = 4
   466:     proj_hidden_dim = 2048
   467:     proj_output_dim = 2048
   468:     use_amp = True
   469:     dtype = torch.bfloat16
   470:     log_every = 10
   471: 
   472:     # Setup
   473:     seed_everything(seed)
   474: 
   475:     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   476:     print(f"Using device: {device}", flush=True)
   477: 
   478:     # Data
   479:     base_train_dataset = CIFAR10(
   480:         root=data_dir, train=True, download=False, transform=None
   481:     )
   482:     train_dataset = ImageDataset(base_train_dataset, get_train_transforms(), num_crops=2)
   483:     val_dataset = CIFAR10(
   484:         root=data_dir, train=False, download=False, transform=get_val_transforms()
   485:     )
   486: 
   487:     train_loader = DataLoader(
   488:         train_dataset,
   489:         batch_size=batch_size,
   490:         shuffle=True,
   491:         num_workers=num_workers,
   492:         pin_memory=True,
   493:         drop_last=True,
   494:         worker_init_fn=seed_worker,
   495:         generator=make_generator(seed),
   496:     )
   497:     val_loader = DataLoader(
   498:         val_dataset,
   499:         batch_size=batch_size,
   500:         shuffle=False,

[truncated: showing at most 500 lines / 60000 bytes from eb_jepa/custom_regularizer.py]
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **resnet18** — wall-clock budget `12:00:00`, compute share `0.33`
- **resnet34** — wall-clock budget `12:00:00`, compute share `0.33`
- **resnet50** — wall-clock budget `12:00:00`, compute share `0.33`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `naive` baseline — editable region  [READ-ONLY — reference implementation]

In `eb_jepa/custom_regularizer.py`:

```python
Lines 33–46:
    30: 
    31: # ── Custom Regularizer ─────────────────────────────────────────────────────
    32: # EDITABLE REGION START
    33: class CustomRegularizer(nn.Module):
    34:     """Naive MSE-only regularizer (no anti-collapse). Lower-bound baseline."""
    35: 
    36:     def __init__(self):
    37:         super().__init__()
    38: 
    39:     def forward(self, z1, z2):
    40:         loss = F.mse_loss(z1, z2)
    41:         return {"loss": loss, "invariance_loss": loss}
    42: 
    43: 
    44: # CONFIG_OVERRIDES: override training hyperparameters for your method.
    45: # Allowed keys: proj_output_dim, proj_hidden_dim.
    46: CONFIG_OVERRIDES = {}
    47: # EDITABLE REGION END
    48: 
    49: 
```

### `vicreg` baseline — editable region  [READ-ONLY — reference implementation]

In `eb_jepa/custom_regularizer.py`:

```python
Lines 33–75:
    30: 
    31: # ── Custom Regularizer ─────────────────────────────────────────────────────
    32: # EDITABLE REGION START
    33: class CustomRegularizer(nn.Module):
    34:     """VICReg: Variance-Invariance-Covariance Regularization."""
    35: 
    36:     def __init__(self, std_coeff=1.0, cov_coeff=100.0, std_margin=1.0):
    37:         super().__init__()
    38:         self.std_coeff = std_coeff
    39:         self.cov_coeff = cov_coeff
    40:         self.std_margin = std_margin
    41: 
    42:     def _std_loss(self, x):
    43:         x = x - x.mean(dim=0, keepdim=True)
    44:         std = torch.sqrt(x.var(dim=0) + 0.0001)
    45:         return torch.mean(F.relu(self.std_margin - std))
    46: 
    47:     def _off_diagonal(self, x):
    48:         n, m = x.shape
    49:         assert n == m
    50:         return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    51: 
    52:     def _cov_loss(self, x):
    53:         batch_size = x.shape[0]
    54:         x = x - x.mean(dim=0, keepdim=True)
    55:         cov = (x.T @ x) / (batch_size - 1)
    56:         return self._off_diagonal(cov).pow(2).mean()
    57: 
    58:     def forward(self, z1, z2):
    59:         sim_loss = F.mse_loss(z1, z2)
    60:         var_loss = self._std_loss(z1) + self._std_loss(z2)
    61:         cov_loss = self._cov_loss(z1) + self._cov_loss(z2)
    62:         total_loss = sim_loss + self.std_coeff * var_loss + self.cov_coeff * cov_loss
    63:         return {
    64:             "loss": total_loss,
    65:             "invariance_loss": sim_loss,
    66:             "var_loss": var_loss,
    67:             "cov_loss": cov_loss,
    68:         }
    69: 
    70: 
    71: # CONFIG_OVERRIDES: override training hyperparameters for your method.
    72: # Allowed keys: proj_output_dim, proj_hidden_dim.
    73: # Paper README "Impact of the projector" table ranks VICReg's best
    74: # projector as 2048->1024 (90.12% on CIFAR-10 ResNet-18, 300 epochs).
    75: CONFIG_OVERRIDES = {"proj_output_dim": 1024}
    76: # EDITABLE REGION END
    77: 
    78: 
```

### `sigreg` baseline — editable region  [READ-ONLY — reference implementation]

In `eb_jepa/custom_regularizer.py`:

```python
Lines 33–77:
    30: 
    31: # ── Custom Regularizer ─────────────────────────────────────────────────────
    32: # EDITABLE REGION START
    33: class CustomRegularizer(nn.Module):
    34:     """BCS (Batched Characteristic Slicing) regularizer for SIGReg."""
    35: 
    36:     def __init__(self, num_slices=256, lmbd=10.0):
    37:         super().__init__()
    38:         self.num_slices = num_slices
    39:         self.step = 0
    40:         self.lmbd = lmbd
    41: 
    42:     def _epps_pulley(self, x, t_min=-3, t_max=3, n_points=10):
    43:         t = torch.linspace(t_min, t_max, n_points, device=x.device)
    44:         exp_f = torch.exp(-0.5 * t ** 2)
    45:         x_t = x.unsqueeze(2) * t
    46:         ecf = (1j * x_t).exp().mean(0)
    47:         err = exp_f * (ecf - exp_f).abs() ** 2
    48:         T = torch.trapz(err, t, dim=1)
    49:         return T
    50: 
    51:     def forward(self, z1, z2):
    52:         dev = z1.device
    53:         with torch.no_grad():
    54:             g = torch.Generator(device=dev)
    55:             g.manual_seed(self.step)
    56:             proj_shape = (z1.size(1), self.num_slices)
    57:             A = torch.randn(proj_shape, device=dev, generator=g)
    58:             A = A / A.norm(p=2, dim=0)
    59:         view1 = z1 @ A
    60:         view2 = z2 @ A
    61: 
    62:         self.step += 1
    63:         bcs = (self._epps_pulley(view1).mean() + self._epps_pulley(view2).mean()) / 2
    64:         invariance_loss = F.mse_loss(z1, z2)
    65:         total_loss = invariance_loss + self.lmbd * bcs
    66:         return {
    67:             "loss": total_loss,
    68:             "bcs_loss": bcs,
    69:             "invariance_loss": invariance_loss,
    70:         }
    71: 
    72: 
    73: # CONFIG_OVERRIDES: override training hyperparameters for your method.
    74: # Allowed keys: proj_output_dim, proj_hidden_dim.
    75: # Paper sigreg.yaml uses 2048->128 — SIGReg's Gaussianity test on random
    76: # projections concentrates better at low output dims (paper rank-1: 91.02%).
    77: CONFIG_OVERRIDES = {"proj_output_dim": 128}
    78: # EDITABLE REGION END
    79: 
    80: 
```

### `barlow_twins` baseline — editable region  [READ-ONLY — reference implementation]

In `eb_jepa/custom_regularizer.py`:

```python
Lines 33–91:
    30: 
    31: # ── Custom Regularizer ─────────────────────────────────────────────────────
    32: # EDITABLE REGION START
    33: class CustomRegularizer(nn.Module):
    34:     """Barlow Twins (Zbontar et al. ICML 2021).
    35: 
    36:     NB on scale_loss: the paper's official 8192-projector recipe includes
    37:     a `--scale-loss 0.024` multiplier — see the README of the original
    38:     repo's mirror (xuChenSJTU/barlowtwins-1) and solo-learn's reference
    39:     implementation
    40:         https://github.com/vturrisi/solo-learn/blob/main/solo/losses/barlow.py
    41:     Without it the raw loss is on the order of 1e3-1e4, and LARS' adaptive
    42:     rescaling (lars_lr = p_norm / (g_norm + ...)) starves the optimizer
    43:     so the diagonal of the cross-correlation matrix never approaches 1.
    44:     Using paper-default scale_loss=0.024 with the 8192 projector.
    45:     """
    46: 
    47:     def __init__(self, lambd=0.0051, scale_loss=0.1):
    48:         super().__init__()
    49:         self.lambd = lambd
    50:         self.scale_loss = scale_loss
    51:         # Use LazyBatchNorm1d so the module is registered in __init__
    52:         # (with proper to(device)/dtype propagation) but the feature dim
    53:         # is materialized on the first forward call.
    54:         self.bn = nn.LazyBatchNorm1d(affine=False)
    55: 
    56:     @staticmethod
    57:     def _off_diagonal(x):
    58:         # Verbatim from barlowtwins/main.py.
    59:         n, m = x.shape
    60:         assert n == m
    61:         return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    62: 
    63:     def forward(self, z1, z2):
    64:         B = z1.shape[0]
    65: 
    66:         # Cross-correlation matrix (paper forward, verbatim).
    67:         c = self.bn(z1).T @ self.bn(z2)
    68:         c = c / B
    69: 
    70:         on_diag = (torch.diagonal(c) - 1).pow(2).sum()
    71:         off_diag = self._off_diagonal(c).pow(2).sum()
    72:         total_loss = self.scale_loss * (on_diag + self.lambd * off_diag)
    73: 
    74:         return {
    75:             "loss": total_loss,
    76:             "on_diag": on_diag,
    77:             "off_diag": off_diag,
    78:         }
    79: 
    80: 
    81: # CONFIG_OVERRIDES: override training hyperparameters for your method.
    82: # Allowed keys: proj_output_dim, proj_hidden_dim.
    83: # Use the solo-learn CIFAR-10 Barlow Twins recipe (proj=2048,
    84: # scale_loss=0.1) instead of the paper's ImageNet recipe
    85: # (proj=8192, scale_loss=0.024, batch=2048, epochs=1000). Our setup
    86: # matches solo-learn's: CIFAR-10, batch=256, ResNet-{18,34,50}, LARS
    87: # with eta=0.02 and clip_lr=True. The paper's 8192 recipe needs
    88: # epochs=1000 + batch=2048 to converge — at our 100-epoch budget it
    89: # leaves the diagonal stuck (see logs from v3: rn34 only reaches 10%).
    90: # https://github.com/vturrisi/solo-learn/blob/main/scripts/pretrain/cifar/barlow.yaml
    91: CONFIG_OVERRIDES = {}
    92: # EDITABLE REGION END
    93: 
    94: 
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
