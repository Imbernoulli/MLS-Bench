# MLS-Bench: optimization-dp-sgd

# Differentially Private SGD: Privacy-Utility Optimization

## Research Question
Design an improved DP-SGD variant that achieves better privacy-utility tradeoff — higher test accuracy under the same `(epsilon, delta)`-differential privacy budget.

## Background
Differentially Private Stochastic Gradient Descent (DP-SGD) was introduced in Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016; arXiv:1607.00133). The mechanism has two steps: (1) clip each per-sample gradient to a fixed `L2`-norm `C`, and (2) add Gaussian noise of scale `σC` to the aggregated gradient before the optimizer step. The noise multiplier `σ` is calibrated to the desired `(ε, δ)` budget via the moments accountant or RDP/PRV accountants.

A constant clipping threshold and constant noise schedule are suboptimal: gradient magnitudes evolve during training, so a fixed threshold either over-clips (losing useful signal) or under-clips (adding excess noise relative to the post-clip norm), and uniform noise allocation ignores varying gradient informativeness across stages. Recent work explores adaptive clipping (Andrew et al., NeurIPS 2021; arXiv:1905.03871), automatic per-sample clipping (Bu et al., "Automatic Clipping", NeurIPS 2023), and noise-decay schedules.

## Task
Modify the `DPMechanism` class in `custom_dpsgd.py`. Your mechanism receives per-sample gradients and must return aggregated noised gradients. You control gradient clipping strategy, noise calibration, and any per-step adaptation.

## Interface
```python
class DPMechanism:
    def __init__(self, max_grad_norm, noise_multiplier, n_params,
                 dataset_size, batch_size, epochs, target_epsilon, target_delta):
        ...

    def clip_and_noise(self, per_sample_grads, step, epoch) -> list[Tensor]:
        # per_sample_grads: list of tensors [B, *param_shape]
        # Returns: list of noised gradients [*param_shape]
        ...

    def get_effective_sigma(self, step, epoch) -> float:
        # Returns current noise multiplier for privacy accounting
        ...
```

## Constraints
- The total privacy budget `(target_epsilon, target_delta)` is FIXED and checked externally.
- The model architecture, data pipeline, optimizer, and training loop are FIXED.
- Focus on algorithmic innovation in the DP mechanism: clipping strategies, noise schedules, gradient processing.
- Available imports: `torch`, `math`, `numpy` (via the FIXED section), `scipy.optimize`.

## Baselines (paper-cited reference implementations)
- **standard_dpsgd** — Abadi et al. (CCS 2016; arXiv:1607.00133): fixed `C` and constant `σ` calibrated up-front.
- **automatic_clipping** — Bu, Wang, Zha, and Karypis, "Automatic Clipping: Differentially Private Deep Learning Made Easier and Stronger" (NeurIPS 2023; arXiv:2206.07136): per-sample normalization removes the clipping-norm hyperparameter.
- **adaptive_clipping** — Andrew, Thakkar, McMahan, and Ramaswamy, "Differentially Private Learning with Adaptive Clipping" (NeurIPS 2021; arXiv:1905.03871): track an online private quantile of the per-sample norm.
- **noise_decay** — schedule the noise multiplier downward as training proceeds, accounting for the full schedule with the same target `(ε, δ)`.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/opacus/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `opacus/custom_dpsgd.py`
- editable lines **152–233**




## Readable Context


### `opacus/custom_dpsgd.py`  [EDITABLE — lines 152–233 only]

```python
     1: #!/usr/bin/env python3
     2: """DP-SGD benchmark for MLS-Bench: Differentially Private Stochastic Gradient Descent.
     3: 
     4: FIXED sections: model architecture, data loading, privacy accounting, evaluation loop.
     5: EDITABLE section: DPMechanism class — gradient clipping strategy, noise calibration,
     6:                   and per-step privacy mechanism modifications.
     7: 
     8: The agent must implement a DPMechanism that achieves better privacy-utility tradeoff
     9: than standard DP-SGD while respecting the same total privacy budget (epsilon, delta).
    10: """
    11: import argparse
    12: import math
    13: import os
    14: import sys
    15: 
    16: import numpy as np
    17: import torch
    18: import torch.nn as nn
    19: import torch.nn.functional as F
    20: import torch.optim as optim
    21: from scipy import optimize as sp_optimize
    22: from torch.utils.data import DataLoader, Subset
    23: from torchvision import datasets, transforms
    24: 
    25: # =====================================================================
    26: # FIXED: Model architectures (DO NOT MODIFY)
    27: # =====================================================================
    28: 
    29: class MNISTNet(nn.Module):
    30:     """Small ConvNet for MNIST / Fashion-MNIST (1-channel 28x28 images)."""
    31:     def __init__(self):
    32:         super().__init__()
    33:         self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
    34:         self.conv2 = nn.Conv2d(16, 32, 4, 2)
    35:         self.fc1 = nn.Linear(32 * 4 * 4, 32)
    36:         self.fc2 = nn.Linear(32, 10)
    37: 
    38:     def forward(self, x):
    39:         x = F.relu(self.conv1(x))
    40:         x = F.max_pool2d(x, 2, 1)
    41:         x = F.relu(self.conv2(x))
    42:         x = F.max_pool2d(x, 2, 1)
    43:         x = x.view(-1, 32 * 4 * 4)
    44:         x = F.relu(self.fc1(x))
    45:         x = self.fc2(x)
    46:         return x
    47: 
    48: 
    49: class CIFAR10Net(nn.Module):
    50:     """ConvNet for CIFAR-10 (3-channel 32x32 images), using GroupNorm (DP-compatible)."""
    51:     def __init__(self):
    52:         super().__init__()
    53:         self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
    54:         self.gn1 = nn.GroupNorm(8, 32)
    55:         self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
    56:         self.gn2 = nn.GroupNorm(8, 64)
    57:         self.conv3 = nn.Conv2d(64, 64, 3, 1, padding=1)
    58:         self.gn3 = nn.GroupNorm(8, 64)
    59:         self.conv4 = nn.Conv2d(64, 128, 3, 1, padding=1)
    60:         self.gn4 = nn.GroupNorm(8, 128)
    61:         self.fc = nn.Linear(128, 10)
    62: 
    63:     def forward(self, x):
    64:         x = F.relu(self.gn1(self.conv1(x)))
    65:         x = F.avg_pool2d(x, 2, 2)
    66:         x = F.relu(self.gn2(self.conv2(x)))
    67:         x = F.avg_pool2d(x, 2, 2)
    68:         x = F.relu(self.gn3(self.conv3(x)))
    69:         x = F.avg_pool2d(x, 2, 2)
    70:         x = F.relu(self.gn4(self.conv4(x)))
    71:         x = F.adaptive_avg_pool2d(x, (1, 1))
    72:         x = x.view(x.size(0), -1)
    73:         x = self.fc(x)
    74:         return x
    75: 
    76: 
    77: # =====================================================================
    78: # FIXED: Privacy accounting utilities (DO NOT MODIFY)
    79: # =====================================================================
    80: 
    81: def _compute_rdp_single_epoch(q, sigma, alpha):
    82:     """Compute RDP for a single epoch of subsampled Gaussian mechanism."""
    83:     if sigma == 0:
    84:         return float("inf")
    85:     if q == 0:
    86:         return 0.0
    87:     if alpha == 1:
    88:         return q * q / (2 * sigma * sigma)
    89:     log_term = (
    90:         math.lgamma(alpha + 1)
    91:         - math.lgamma(alpha - 1 + 1)
    92:         - math.lgamma(2)
    93:         + (alpha - 1) * math.log(1 - q)
    94:         + math.log(q * q * alpha / (2 * sigma * sigma))
    95:     )
    96:     # Simplified RDP bound for subsampled Gaussian
    97:     return min(
    98:         alpha * q * q / (2 * sigma * sigma),
    99:         q * q * alpha / (2 * sigma * sigma) + q * q * q * alpha * (alpha - 1) / (6 * sigma * sigma),
   100:     )
   101: 
   102: 
   103: def compute_epsilon(steps, sigma, q, delta, alphas=None):
   104:     """Compute (epsilon, best_alpha) via RDP accounting.
   105: 
   106:     Args:
   107:         steps: number of training steps
   108:         sigma: noise multiplier
   109:         q: sampling probability (batch_size / dataset_size)
   110:         delta: target delta
   111:         alphas: list of RDP orders to try
   112: 
   113:     Returns:
   114:         (epsilon, best_alpha)
   115:     """
   116:     if alphas is None:
   117:         alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
   118:     best_eps = float("inf")
   119:     best_alpha = None
   120:     for alpha in alphas:
   121:         # RDP for subsampled Gaussian mechanism (tight bound)
   122:         if alpha <= 1:
   123:             continue
   124:         rdp = steps * min(
   125:             q * q * alpha / (2 * sigma * sigma),
   126:             alpha * q * q / (2 * sigma * sigma),
   127:         )
   128:         # Convert RDP to (epsilon, delta)-DP
   129:         eps = rdp - math.log(delta) / (alpha - 1) + math.log(1 - 1 / alpha)
   130:         if eps < best_eps:
   131:             best_eps = eps
   132:             best_alpha = alpha
   133:     return max(0, best_eps), best_alpha
   134: 
   135: 
   136: def calibrate_noise_to_epsilon(target_epsilon, steps, q, delta, tol=1e-3):
   137:     """Find the noise multiplier sigma that achieves target_epsilon.
   138: 
   139:     Uses binary search to find the right noise level.
   140:     """
   141:     sigma_low, sigma_high = 0.01, 100.0
   142:     while sigma_high - sigma_low > tol:
   143:         sigma_mid = (sigma_low + sigma_high) / 2
   144:         eps, _ = compute_epsilon(steps, sigma_mid, q, delta)
   145:         if eps > target_epsilon:
   146:             sigma_low = sigma_mid
   147:         else:
   148:             sigma_high = sigma_mid
   149:     return (sigma_low + sigma_high) / 2
   150: 
   151: 
   152: # =====================================================================
   153: # EDITABLE SECTION START (lines 152-233)
   154: # =====================================================================
   155: # DPMechanism: Controls how per-sample gradients are clipped and noised.
   156: #
   157: # Interface contract:
   158: #   __init__(self, max_grad_norm, noise_multiplier, n_params, dataset_size,
   159: #            batch_size, epochs, target_epsilon, target_delta)
   160: #   clip_and_noise(self, per_sample_grads, step, epoch) -> noised_gradient
   161: #   get_effective_sigma(self, step, epoch) -> float
   162: #
   163: # The mechanism receives per-sample gradients (list of tensors, each [B, *param_shape])
   164: # and must return aggregated + noised gradients (list of tensors, each [*param_shape]).
   165: #
   166: # IMPORTANT:
   167: # - The total privacy budget (target_epsilon, target_delta) is FIXED.
   168: # - Your mechanism must not exceed it. The accounting is checked externally.
   169: # - You may adapt clipping thresholds, noise schedules, or gradient processing
   170: #   as long as privacy guarantees hold.
   171: 
   172: class DPMechanism:
   173:     """Differentially private gradient mechanism.
   174: 
   175:     Standard DP-SGD: clip per-sample gradients to max_grad_norm,
   176:     then add Gaussian noise calibrated to (noise_multiplier * max_grad_norm).
   177:     """
   178: 
   179:     def __init__(self, max_grad_norm, noise_multiplier, n_params,
   180:                  dataset_size, batch_size, epochs, target_epsilon, target_delta):
   181:         self.max_grad_norm = max_grad_norm
   182:         self.noise_multiplier = noise_multiplier
   183:         self.n_params = n_params
   184:         self.dataset_size = dataset_size
   185:         self.batch_size = batch_size
   186:         self.epochs = epochs
   187:         self.target_epsilon = target_epsilon
   188:         self.target_delta = target_delta
   189: 
   190:     def clip_and_noise(self, per_sample_grads, step, epoch):
   191:         """Clip per-sample gradients and add noise.
   192: 
   193:         Args:
   194:             per_sample_grads: list of tensors, each [B, *param_shape]
   195:             step: current global training step
   196:             epoch: current epoch number
   197: 
   198:         Returns:
   199:             list of noised gradient tensors, each [*param_shape]
   200:         """
   201:         batch_size = per_sample_grads[0].shape[0]
   202: 
   203:         # Compute per-sample gradient norms (flat norm across all parameters)
   204:         flat = torch.cat([g.reshape(batch_size, -1) for g in per_sample_grads], dim=1)
   205:         norms = flat.norm(2, dim=1)  # [B]
   206: 
   207:         # Clip per-sample gradients
   208:         clip_factor = (self.max_grad_norm / norms.clamp(min=1e-8)).clamp(max=1.0)  # [B]
   209: 
   210:         noised_grads = []
   211:         for g in per_sample_grads:
   212:             # Apply clipping: g[i] *= clip_factor[i]
   213:             shape = [batch_size] + [1] * (g.dim() - 1)
   214:             clipped = g * clip_factor.reshape(shape)
   215: 
   216:             # Average over batch
   217:             avg = clipped.mean(dim=0)
   218: 
   219:             # Add calibrated Gaussian noise
   220:             noise = torch.randn_like(avg) * (
   221:                 self.noise_multiplier * self.max_grad_norm / batch_size
   222:             )
   223:             noised_grads.append(avg + noise)
   224: 
   225:         return noised_grads
   226: 
   227:     def get_effective_sigma(self, step, epoch):
   228:         """Return the effective noise multiplier for privacy accounting."""
   229:         return self.noise_multiplier
   230: 
   231: # =====================================================================
   232: # EDITABLE SECTION END
   233: # =====================================================================
   234: 
   235: 
   236: # =====================================================================
   237: # FIXED: Data loading (DO NOT MODIFY)
   238: # =====================================================================
   239: 
   240: def get_data_loaders(dataset_name, batch_size, data_root=os.environ.get("DATA_ROOT", "/data")):
   241:     """Create train and test data loaders."""
   242:     if dataset_name == "mnist":
   243:         transform = transforms.Compose([
   244:             transforms.ToTensor(),
   245:             transforms.Normalize((0.1307,), (0.3081,)),
   246:         ])
   247:         train_ds = datasets.MNIST(
   248:             os.path.join(data_root, "mnist"), train=True, download=False, transform=transform
   249:         )
   250:         test_ds = datasets.MNIST(
   251:             os.path.join(data_root, "mnist"), train=False, download=False, transform=transform
   252:         )
   253:         model_cls = MNISTNet
   254:     elif dataset_name == "cifar10":
   255:         transform_train = transforms.Compose([
   256:             transforms.ToTensor(),
   257:             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
   258:         ])
   259:         transform_test = transforms.Compose([
   260:             transforms.ToTensor(),
   261:             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
   262:         ])
   263:         train_ds = datasets.CIFAR10(
   264:             os.path.join(data_root, "cifar10"), train=True, download=False, transform=transform_train
   265:         )
   266:         test_ds = datasets.CIFAR10(
   267:             os.path.join(data_root, "cifar10"), train=False, download=False, transform=transform_test
   268:         )
   269:         model_cls = CIFAR10Net
   270:     elif dataset_name == "fmnist":
   271:         transform = transforms.Compose([
   272:             transforms.ToTensor(),
   273:             transforms.Normalize((0.2860,), (0.3530,)),
   274:         ])
   275:         train_ds = datasets.FashionMNIST(
   276:             os.path.join(data_root, "fmnist"), train=True, download=False, transform=transform
   277:         )
   278:         test_ds = datasets.FashionMNIST(
   279:             os.path.join(data_root, "fmnist"), train=False, download=False, transform=transform
   280:         )
   281:         model_cls = MNISTNet
   282:     else:
   283:         raise ValueError(f"Unknown dataset: {dataset_name}")
   284: 
   285:     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
   286:                               num_workers=2, pin_memory=True, drop_last=True)
   287:     test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False,
   288:                              num_workers=2, pin_memory=True)
   289:     return train_ds, train_loader, test_loader, model_cls
   290: 
   291: 
   292: # =====================================================================
   293: # FIXED: Per-sample gradient computation (DO NOT MODIFY)
   294: # =====================================================================
   295: 
   296: def compute_per_sample_gradients(model, data, target, criterion):
   297:     """Compute per-sample gradients using functorch-style vmap.
   298: 
   299:     Returns a list of tensors, each of shape [B, *param_shape].
   300:     """
   301:     params = [p for p in model.parameters() if p.requires_grad]
   302: 
   303:     # Manual per-sample gradient computation via backward on each sample
   304:     batch_size = data.shape[0]
   305:     per_sample_grads = [torch.zeros(batch_size, *p.shape, device=p.device) for p in params]
   306: 
   307:     for i in range(batch_size):
   308:         model.zero_grad()
   309:         output = model(data[i:i+1])
   310:         loss = criterion(output, target[i:i+1])
   311:         loss.backward()
   312:         for j, p in enumerate(params):
   313:             if p.grad is not None:
   314:                 per_sample_grads[j][i] = p.grad.clone()
   315: 
   316:     return per_sample_grads
   317: 
   318: 
   319: def compute_per_sample_gradients_fast(model, data, target, criterion):
   320:     """Efficient per-sample gradient computation using ghost clipping trick.
   321: 
   322:     Computes per-sample gradient norms first, then uses weighted loss for aggregation.
   323:     Falls back to loop-based computation for small batches.
   324:     """
   325:     batch_size = data.shape[0]
   326: 
   327:     # For moderate batch sizes, use vectorized approach via autograd
   328:     if batch_size <= 128:
   329:         return compute_per_sample_gradients(model, data, target, criterion)
   330: 
   331:     # For larger batches, use microbatching for memory efficiency
   332:     micro_bs = 64
   333:     params = [p for p in model.parameters() if p.requires_grad]
   334:     per_sample_grads = [torch.zeros(batch_size, *p.shape, device=p.device) for p in params]
   335: 
   336:     for start in range(0, batch_size, micro_bs):
   337:         end = min(start + micro_bs, batch_size)
   338:         micro_data = data[start:end]
   339:         micro_target = target[start:end]
   340:         for i in range(end - start):
   341:             model.zero_grad()
   342:             output = model(micro_data[i:i+1])
   343:             loss = criterion(output, micro_target[i:i+1])
   344:             loss.backward()
   345:             for j, p in enumerate(params):
   346:                 if p.grad is not None:
   347:                     per_sample_grads[j][start + i] = p.grad.clone()
   348: 
   349:     return per_sample_grads
   350: 
   351: 
   352: # =====================================================================
   353: # FIXED: Training and evaluation loops (DO NOT MODIFY)
   354: # =====================================================================
   355: 
   356: def train_epoch(model, train_loader, optimizer, criterion, dp_mechanism, device,
   357:                 epoch, total_steps, log_interval=50):
   358:     """Train one epoch with DP mechanism."""
   359:     model.train()
   360:     running_loss = 0.0
   361:     correct = 0
   362:     total = 0
   363:     step = total_steps
   364: 
   365:     for batch_idx, (data, target) in enumerate(train_loader):
   366:         data, target = data.to(device), target.to(device)
   367:         batch_size = data.shape[0]
   368: 
   369:         # Compute per-sample gradients
   370:         per_sample_grads = compute_per_sample_gradients(model, data, target, criterion)
   371: 
   372:         # Apply DP mechanism (EDITABLE part)
   373:         noised_grads = dp_mechanism.clip_and_noise(per_sample_grads, step, epoch)
   374: 
   375:         # Set model gradients
   376:         optimizer.zero_grad()
   377:         for param, grad in zip(
   378:             [p for p in model.parameters() if p.requires_grad], noised_grads
   379:         ):
   380:             param.grad = grad
   381: 
   382:         optimizer.step()
   383: 
   384:         # Compute batch metrics (without grad)
   385:         with torch.no_grad():
   386:             output = model(data)
   387:             loss = criterion(output, target)
   388:             running_loss += loss.item() * batch_size
   389:             pred = output.argmax(dim=1)
   390:             correct += pred.eq(target).sum().item()
   391:             total += batch_size
   392: 
   393:         step += 1
   394: 
   395:         if (batch_idx + 1) % log_interval == 0:
   396:             avg_loss = running_loss / total
   397:             acc = 100.0 * correct / total
   398:             print(
   399:                 f"TRAIN_METRICS epoch={epoch} step={step} loss={avg_loss:.6f} "
   400:                 f"accuracy={acc:.2f}",
   401:                 flush=True,
   402:             )
   403: 
   404:     return step, running_loss / total, 100.0 * correct / total
   405: 
   406: 
   407: def evaluate(model, test_loader, criterion, device):
   408:     """Evaluate model on test set."""
   409:     model.eval()
   410:     test_loss = 0.0
   411:     correct = 0
   412:     total = 0
   413: 
   414:     with torch.no_grad():
   415:         for data, target in test_loader:
   416:             data, target = data.to(device), target.to(device)
   417:             output = model(data)
   418:             test_loss += criterion(output, target).item() * data.shape[0]
   419:             pred = output.argmax(dim=1)
   420:             correct += pred.eq(target).sum().item()
   421:             total += data.shape[0]
   422: 
   423:     return test_loss / total, 100.0 * correct / total
   424: 
   425: 
   426: # =====================================================================
   427: # FIXED: Main entry point (DO NOT MODIFY)
   428: # =====================================================================
   429: 
   430: def main():
   431:     parser = argparse.ArgumentParser(description="DP-SGD Benchmark")
   432:     parser.add_argument("--dataset", type=str, default="mnist",
   433:                         choices=["mnist", "cifar10", "fmnist"],
   434:                         help="Dataset to train on")
   435:     parser.add_argument("--epochs", type=int, default=20,
   436:                         help="Number of training epochs")
   437:     parser.add_argument("--batch-size", type=int, default=256,
   438:                         help="Training batch size")
   439:     parser.add_argument("--lr", type=float, default=0.1,
   440:                         help="Learning rate")
   441:     parser.add_argument("--max-grad-norm", type=float, default=1.0,
   442:                         help="Max per-sample gradient norm for clipping")
   443:     parser.add_argument("--target-epsilon", type=float, default=3.0,
   444:                         help="Target epsilon for privacy budget")
   445:     parser.add_argument("--target-delta", type=float, default=1e-5,
   446:                         help="Target delta for privacy budget")
   447:     parser.add_argument("--seed", type=int, default=42,
   448:                         help="Random seed")
   449:     parser.add_argument("--device", type=str, default="cuda",
   450:                         help="Device to use")
   451:     args = parser.parse_args()
   452: 
   453:     # Set seeds
   454:     torch.manual_seed(args.seed)
   455:     np.random.seed(args.seed)
   456:     if torch.cuda.is_available():
   457:         torch.cuda.manual_seed_all(args.seed)
   458: 
   459:     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
   460: 
   461:     # Load data
   462:     train_ds, train_loader, test_loader, model_cls = get_data_loaders(
   463:         args.dataset, args.batch_size
   464:     )
   465:     dataset_size = len(train_ds)
   466:     q = args.batch_size / dataset_size
   467:     steps_per_epoch = len(train_loader)
   468:     total_steps = steps_per_epoch * args.epochs
   469: 
   470:     # Calibrate noise to target epsilon
   471:     sigma = calibrate_noise_to_epsilon(
   472:         args.target_epsilon, total_steps, q, args.target_delta
   473:     )
   474:     print(f"Calibrated noise_multiplier sigma={sigma:.4f} for "
   475:           f"epsilon={args.target_epsilon}, delta={args.target_delta}, "
   476:           f"steps={total_steps}, q={q:.4f}", flush=True)
   477: 
   478:     # Create model
   479:     model = model_cls().to(device)
   480:     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
   481:     print(f"Model: {model_cls.__name__}, Parameters: {n_params}", flush=True)
   482: 
   483:     # Create optimizer (SGD with momentum, standard for DP training)
   484:     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
   485: 
   486:     # Learning rate schedule: cosine annealing
   487:     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
   488: 
   489:     criterion = nn.CrossEntropyLoss()
   490: 
   491:     # Initialize DP mechanism (EDITABLE)
   492:     dp_mechanism = DPMechanism(
   493:         max_grad_norm=args.max_grad_norm,
   494:         noise_multiplier=sigma,
   495:         n_params=n_params,
   496:         dataset_size=dataset_size,
   497:         batch_size=args.batch_size,
   498:         epochs=args.epochs,
   499:         target_epsilon=args.target_epsilon,
   500:         target_delta=args.target_delta,

[truncated: showing at most 500 lines / 60000 bytes from opacus/custom_dpsgd.py]
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `standard_dpsgd` baseline — editable region  [READ-ONLY — reference implementation]

In `opacus/custom_dpsgd.py`:

```python
Lines 152–196:
   149:     return (sigma_low + sigma_high) / 2
   150: 
   151: 
   152: class DPMechanism:
   153:     """Standard DP-SGD (Abadi et al., 2016).
   154: 
   155:     Fixed per-sample gradient clipping + constant Gaussian noise.
   156:     """
   157: 
   158:     def __init__(self, max_grad_norm, noise_multiplier, n_params,
   159:                  dataset_size, batch_size, epochs, target_epsilon, target_delta):
   160:         self.max_grad_norm = max_grad_norm
   161:         self.noise_multiplier = noise_multiplier
   162:         self.n_params = n_params
   163:         self.dataset_size = dataset_size
   164:         self.batch_size = batch_size
   165:         self.epochs = epochs
   166:         self.target_epsilon = target_epsilon
   167:         self.target_delta = target_delta
   168: 
   169:     def clip_and_noise(self, per_sample_grads, step, epoch):
   170:         batch_size = per_sample_grads[0].shape[0]
   171: 
   172:         # Compute per-sample gradient norms (flat norm across all parameters)
   173:         flat = torch.cat([g.reshape(batch_size, -1) for g in per_sample_grads], dim=1)
   174:         norms = flat.norm(2, dim=1)  # [B]
   175: 
   176:         # Clip per-sample gradients
   177:         clip_factor = (self.max_grad_norm / norms.clamp(min=1e-8)).clamp(max=1.0)  # [B]
   178: 
   179:         noised_grads = []
   180:         for g in per_sample_grads:
   181:             shape = [batch_size] + [1] * (g.dim() - 1)
   182:             clipped = g * clip_factor.reshape(shape)
   183: 
   184:             # Average over batch
   185:             avg = clipped.mean(dim=0)
   186: 
   187:             # Add calibrated Gaussian noise
   188:             noise = torch.randn_like(avg) * (
   189:                 self.noise_multiplier * self.max_grad_norm / batch_size
   190:             )
   191:             noised_grads.append(avg + noise)
   192: 
   193:         return noised_grads
   194: 
   195:     def get_effective_sigma(self, step, epoch):
   196:         return self.noise_multiplier
   197: 
   198: 
   199: # =====================================================================
```

### `automatic_clipping` baseline — editable region  [READ-ONLY — reference implementation]

In `opacus/custom_dpsgd.py`:

```python
Lines 152–202:
   149:     return (sigma_low + sigma_high) / 2
   150: 
   151: 
   152: class DPMechanism:
   153:     """AUTO-S Automatic Clipping (Bu et al., NeurIPS 2023).
   154: 
   155:     Per-sample gradient normalization: g_i / (||g_i|| + gamma).
   156:     Sensitivity bounded by 1, no clipping threshold to tune.
   157:     """
   158: 
   159:     def __init__(self, max_grad_norm, noise_multiplier, n_params,
   160:                  dataset_size, batch_size, epochs, target_epsilon, target_delta):
   161:         self.max_grad_norm = max_grad_norm
   162:         self.noise_multiplier = noise_multiplier
   163:         self.n_params = n_params
   164:         self.dataset_size = dataset_size
   165:         self.batch_size = batch_size
   166:         self.epochs = epochs
   167:         self.target_epsilon = target_epsilon
   168:         self.target_delta = target_delta
   169:         # AUTO-S gamma for this benchmark harness. gamma=1.0 keeps the
   170:         # existing learning-rate schedule stable.
   171:         self.gamma = 1.0
   172: 
   173:     def clip_and_noise(self, per_sample_grads, step, epoch):
   174:         batch_size = per_sample_grads[0].shape[0]
   175: 
   176:         # Compute per-sample gradient norms
   177:         flat = torch.cat([g.reshape(batch_size, -1) for g in per_sample_grads], dim=1)
   178:         norms = flat.norm(2, dim=1)  # [B]
   179: 
   180:         # AUTO-S normalization: scale each gradient by 1/(||g_i|| + gamma)
   181:         # This bounds sensitivity to 1 (since ||g_i / (||g_i|| + gamma)|| <= 1)
   182:         scale = 1.0 / (norms + self.gamma)  # [B]
   183: 
   184:         noised_grads = []
   185:         for g in per_sample_grads:
   186:             shape = [batch_size] + [1] * (g.dim() - 1)
   187:             normalized = g * scale.reshape(shape)
   188: 
   189:             # Average over batch
   190:             avg = normalized.mean(dim=0)
   191: 
   192:             # Add noise calibrated to sensitivity=1 (AUTO-S bound)
   193:             # sigma * C / B where C=1 for AUTO-S
   194:             noise = torch.randn_like(avg) * (
   195:                 self.noise_multiplier * 1.0 / batch_size
   196:             )
   197:             noised_grads.append(avg + noise)
   198: 
   199:         return noised_grads
   200: 
   201:     def get_effective_sigma(self, step, epoch):
   202:         return self.noise_multiplier
   203: 
   204: 
   205: # =====================================================================
```

### `adaptive_clipping` baseline — editable region  [READ-ONLY — reference implementation]

In `opacus/custom_dpsgd.py`:

```python
Lines 152–213:
   149:     return (sigma_low + sigma_high) / 2
   150: 
   151: 
   152: class DPMechanism:
   153:     """Adaptive Quantile Clipping (Andrew et al., NeurIPS 2021).
   154: 
   155:     Dynamically adjusts clipping threshold to target quantile of gradient norms.
   156:     """
   157: 
   158:     def __init__(self, max_grad_norm, noise_multiplier, n_params,
   159:                  dataset_size, batch_size, epochs, target_epsilon, target_delta):
   160:         self.max_grad_norm = max_grad_norm
   161:         self.noise_multiplier = noise_multiplier
   162:         self.n_params = n_params
   163:         self.dataset_size = dataset_size
   164:         self.batch_size = batch_size
   165:         self.epochs = epochs
   166:         self.target_epsilon = target_epsilon
   167:         self.target_delta = target_delta
   168: 
   169:         # Adaptive clipping parameters for the Andrew et al. update rule.
   170:         self.clip_norm = max_grad_norm  # Initial clipping threshold
   171:         self.target_quantile = 0.5  # Target: median of gradient norms
   172:         self.clip_lr = 0.2  # Learning rate for clipping threshold adaptation
   173:         self.clip_min = 0.01  # Minimum clipping threshold
   174:         self.clip_max = 100.0  # Maximum clipping threshold
   175: 
   176:     def clip_and_noise(self, per_sample_grads, step, epoch):
   177:         batch_size = per_sample_grads[0].shape[0]
   178: 
   179:         # Compute per-sample gradient norms
   180:         flat = torch.cat([g.reshape(batch_size, -1) for g in per_sample_grads], dim=1)
   181:         norms = flat.norm(2, dim=1)  # [B]
   182: 
   183:         # Compute fraction of samples exceeding current clip norm
   184:         frac_above = (norms > self.clip_norm).float().mean().item()
   185: 
   186:         # Update clipping threshold using geometric update
   187:         # If too many gradients are clipped, increase threshold; if too few, decrease
   188:         self.clip_norm = self.clip_norm * math.exp(
   189:             self.clip_lr * (frac_above - self.target_quantile)
   190:         )
   191:         self.clip_norm = max(self.clip_min, min(self.clip_max, self.clip_norm))
   192: 
   193:         # Clip per-sample gradients using adaptive threshold
   194:         clip_factor = (self.clip_norm / norms.clamp(min=1e-8)).clamp(max=1.0)
   195: 
   196:         noised_grads = []
   197:         for g in per_sample_grads:
   198:             shape = [batch_size] + [1] * (g.dim() - 1)
   199:             clipped = g * clip_factor.reshape(shape)
   200: 
   201:             # Average over batch
   202:             avg = clipped.mean(dim=0)
   203: 
   204:             # Add noise calibrated to current clip norm
   205:             noise = torch.randn_like(avg) * (
   206:                 self.noise_multiplier * self.clip_norm / batch_size
   207:             )
   208:             noised_grads.append(avg + noise)
   209: 
   210:         return noised_grads
   211: 
   212:     def get_effective_sigma(self, step, epoch):
   213:         return self.noise_multiplier
   214: 
   215: 
   216: # =====================================================================
```

### `noise_decay` baseline — editable region  [READ-ONLY — reference implementation]

In `opacus/custom_dpsgd.py`:

```python
Lines 152–268:
   149:     return (sigma_low + sigma_high) / 2
   150: 
   151: 
   152: class DPMechanism:
   153:     """Step-Decay Noise Schedule (inspired by Global-Adapt-V2-S, 2025).
   154: 
   155:     Decays noise multiplier and clipping threshold over training epochs
   156:     to allocate more privacy budget to later (more useful) training steps.
   157: 
   158:     Privacy accounting: tracks cumulative RDP per-step using the actual
   159:     sigma at each step, then returns an equivalent uniform sigma so
   160:     that the external ``compute_epsilon(steps, sigma, q, delta)`` call
   161:     produces the correct (tight) epsilon.
   162:     """
   163: 
   164:     def __init__(self, max_grad_norm, noise_multiplier, n_params,
   165:                  dataset_size, batch_size, epochs, target_epsilon, target_delta):
   166:         self.max_grad_norm = max_grad_norm
   167:         self.noise_multiplier = noise_multiplier
   168:         self.n_params = n_params
   169:         self.dataset_size = dataset_size
   170:         self.batch_size = batch_size
   171:         self.epochs = epochs
   172:         self.target_epsilon = target_epsilon
   173:         self.target_delta = target_delta
   174: 
   175:         # Step-decay schedule parameters
   176:         # Decay noise and clipping every decay_interval epochs
   177:         self.decay_interval = max(1, epochs // 4)  # 4 decay stages
   178:         self.noise_decay_factor = 0.8  # Reduce noise by 20% at each stage
   179:         self.clip_decay_factor = 0.85  # Reduce clip norm by 15% at each stage
   180: 
   181:         # Pre-compute the per-epoch sigma schedule so we can do accurate
   182:         # RDP accounting.  Steps per epoch = dataset_size // batch_size
   183:         # (drop_last=True in DataLoader).
   184:         self.steps_per_epoch = dataset_size // batch_size
   185: 
   186:         # Compute sigma_0: scale the calibrated (uniform) sigma up so that
   187:         # the harmonic-mean-equivalent sigma across all steps equals the
   188:         # calibrated value.  This keeps the total privacy spend equal to
   189:         # the budget even though individual steps have different noise.
   190:         total_steps = self.steps_per_epoch * epochs
   191:         inv_sq_sum = 0.0
   192:         for e in range(1, epochs + 1):
   193:             stage = (e - 1) // self.decay_interval
   194:             factor = self.noise_decay_factor ** stage
   195:             # Each epoch contributes steps_per_epoch steps at sigma_0*factor
   196:             # 1/sigma_t^2 = 1/(sigma_0*factor)^2 = 1/(sigma_0^2 * factor^2)
   197:             inv_sq_sum += self.steps_per_epoch / (factor * factor)
   198:         # sigma_eff = sqrt(total_steps / inv_sq_sum) * sigma_0
   199:         # We want sigma_eff == noise_multiplier (the calibrated value), so:
   200:         #   noise_multiplier = sigma_0 * sqrt(total_steps / inv_sq_sum)
   201:         #   sigma_0 = noise_multiplier / sqrt(total_steps / inv_sq_sum)
   202:         #           = noise_multiplier * sqrt(inv_sq_sum / total_steps)
   203:         self.sigma_0 = noise_multiplier * (inv_sq_sum / total_steps) ** 0.5
   204:         self.clip_0 = max_grad_norm
   205: 
   206:         # Current values
   207:         self._current_sigma = self.sigma_0
   208:         self._current_clip = self.clip_0
   209: 
   210:     def clip_and_noise(self, per_sample_grads, step, epoch):
   211:         batch_size = per_sample_grads[0].shape[0]
   212: 
   213:         # Update schedule based on epoch
   214:         stage = (epoch - 1) // self.decay_interval
   215:         self._current_sigma = self.sigma_0 * (self.noise_decay_factor ** stage)
   216:         self._current_clip = self.clip_0 * (self.clip_decay_factor ** stage)
   217: 
   218:         # Compute per-sample gradient norms
   219:         flat = torch.cat([g.reshape(batch_size, -1) for g in per_sample_grads], dim=1)
   220:         norms = flat.norm(2, dim=1)  # [B]
   221: 
   222:         # Clip per-sample gradients using current (decayed) threshold
   223:         clip_factor = (self._current_clip / norms.clamp(min=1e-8)).clamp(max=1.0)
   224: 
   225:         noised_grads = []
   226:         for g in per_sample_grads:
   227:             shape = [batch_size] + [1] * (g.dim() - 1)
   228:             clipped = g * clip_factor.reshape(shape)
   229: 
   230:             # Average over batch
   231:             avg = clipped.mean(dim=0)
   232: 
   233:             # Add noise calibrated to current clip norm and sigma
   234:             noise = torch.randn_like(avg) * (
   235:                 self._current_sigma * self._current_clip / batch_size
   236:             )
   237:             noised_grads.append(avg + noise)
   238: 
   239:         return noised_grads
   240: 
   241:     def get_effective_sigma(self, step, epoch):
   242:         """Return equivalent uniform sigma for accurate RDP accounting.
   243: 
   244:         Computes the harmonic-mean-equivalent sigma over all steps up to
   245:         the current point, so that the external call
   246:         ``compute_epsilon(step, sigma_eff, q, delta)`` which assumes a
   247:         uniform sigma gives the same epsilon as step-by-step RDP
   248:         accounting with the actual per-step sigma values.
   249: 
   250:         sigma_eff = sqrt(steps / sum_{t=1}^{steps} 1/sigma_t^2)
   251:         """
   252:         if step <= 0:
   253:             return self.sigma_0
   254:         # Accumulate 1/sigma_t^2 across completed steps
   255:         inv_sq_sum = 0.0
   256:         steps_counted = 0
   257:         for e in range(1, self.epochs + 1):
   258:             stage = (e - 1) // self.decay_interval
   259:             sigma_e = self.sigma_0 * (self.noise_decay_factor ** stage)
   260:             inv_sq_e = 1.0 / (sigma_e * sigma_e)
   261:             epoch_steps = min(self.steps_per_epoch, step - steps_counted)
   262:             if epoch_steps <= 0:
   263:                 break
   264:             inv_sq_sum += epoch_steps * inv_sq_e
   265:             steps_counted += epoch_steps
   266:         if inv_sq_sum == 0:
   267:             return self.sigma_0
   268:         return (steps_counted / inv_sq_sum) ** 0.5
   269: 
   270: 
   271: # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
