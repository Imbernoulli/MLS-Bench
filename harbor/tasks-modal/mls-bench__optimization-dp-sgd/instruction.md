# MLS-Bench: optimization-dp-sgd

# Differentially Private SGD: Privacy-Utility Optimization

## Research Question
Design an improved DP-SGD variant that achieves better privacy-utility tradeoff — higher test accuracy under the same `(epsilon, delta)`-differential privacy budget.

## Background
Differentially Private Stochastic Gradient Descent (DP-SGD) was introduced in Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016; arXiv:1607.00133). The mechanism has two steps: (1) clip each per-sample gradient to a fixed `L2`-norm `C`, and (2) add Gaussian noise of scale `σC` to the aggregated gradient before the optimizer step. The noise multiplier `σ` is calibrated to the desired `(ε, δ)` budget via the moments accountant or RDP/PRV accountants.

A constant clipping threshold and constant noise schedule are suboptimal: gradient magnitudes evolve during training, so a fixed threshold either over-clips (losing useful signal) or under-clips (adding excess noise relative to the post-clip norm), and uniform noise allocation ignores varying gradient informativeness across stages. Recent work explores adaptive clipping (Andrew et al., NeurIPS 2021; arXiv:1905.03871), automatic per-sample clipping (Bu et al., "Automatic Clipping", NeurIPS 2023), and noise-decay schedules.

## Task
Modify the `DPMechanism` class in `custom_dpsgd.py`. Your mechanism receives per-sample gradients and decides how each one is clipped (per-sample multipliers and the clipping bound `C_t`) and the noise multiplier `σ_t` of each step. You control gradient clipping strategy, noise calibration, and any per-step adaptation. The FIXED harness applies your multipliers, checks that every clipped per-sample gradient has `L2`-norm at most `C_t`, averages over the batch, adds Gaussian noise of std `σ_t·C_t/B`, and accounts the privacy spent from the `σ_t` it applied.

## Interface
```python
class DPMechanism:
    def __init__(self, max_grad_norm, noise_multiplier, n_params,
                 dataset_size, batch_size, epochs, target_epsilon, target_delta):
        ...

    def clip(self, per_sample_grads, step, epoch) -> tuple[Tensor | list[Tensor], float]:
        # per_sample_grads: copy of the list of tensors [B, *param_shape]
        # Returns (scale, clip_norm): per-sample multipliers, a [B] tensor
        # (or a list of [B] tensors, one per parameter), and the L2 bound C_t
        # that every scaled per-sample gradient satisfies
        ...

    def get_noise_multiplier(self, step, epoch) -> float:
        # Returns this step's noise multiplier sigma_t (called after clip);
        # the harness adds N(0, (sigma_t * C_t / B)^2) noise and accounts it
        ...
```

## Constraints
- The total privacy budget `(target_epsilon, target_delta)` is FIXED and checked externally: the harness composes the per-step `σ_t` it applied, and a run whose epsilon exceeds the target, or whose clipped per-sample gradient exceeds its declared `C_t`, is aborted.
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
(inclusive, 1-indexed)**. Edits that change code outside these ranges — or creating new files, or
deleting whole files — will cause your submission to be invalid.

The line numbers mark an editable **region**, not a fixed line-count budget: you
may add or remove lines inside it. Only code outside the editable ranges must
stay unchanged.

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
   155: # DPMechanism: decides how per-sample gradients are clipped and how much
   156: # noise each step gets. The FIXED harness (`privatize_step`, below) applies
   157: # your per-sample scaling, checks the clipping bound, adds the Gaussian
   158: # noise and does the privacy accounting with the values it actually used.
   159: #
   160: # Interface contract:
   161: #   __init__(self, max_grad_norm, noise_multiplier, n_params, dataset_size,
   162: #            batch_size, epochs, target_epsilon, target_delta)
   163: #   clip(self, per_sample_grads, step, epoch) -> (scale, clip_norm)
   164: #   get_noise_multiplier(self, step, epoch) -> float
   165: #
   166: # clip() receives a copy of the per-sample gradients (list of tensors, each
   167: # [B, *param_shape]) and returns `scale`, the per-sample multipliers (a [B]
   168: # tensor, or a list with one [B] tensor per parameter), and `clip_norm`, the
   169: # L2 bound C_t that every scaled per-sample gradient satisfies. The harness
   170: # averages the scaled gradients over the batch and adds N(0, (sigma_t*C_t/B)^2)
   171: # noise, where sigma_t = get_noise_multiplier(step, epoch) (called after clip).
   172: #
   173: # IMPORTANT:
   174: # - The total privacy budget (target_epsilon, target_delta) is FIXED. The
   175: #   harness accounts every step with the sigma_t it applied and aborts a run
   176: #   whose epsilon exceeds the target; a scaled per-sample gradient whose norm
   177: #   exceeds clip_norm also aborts the run.
   178: 
   179: class DPMechanism:
   180:     """Differentially private gradient mechanism.
   181: 
   182:     Standard DP-SGD: clip per-sample gradients to max_grad_norm and use the
   183:     calibrated constant noise multiplier (noise std = sigma * C / B).
   184:     """
   185: 
   186:     def __init__(self, max_grad_norm, noise_multiplier, n_params,
   187:                  dataset_size, batch_size, epochs, target_epsilon, target_delta):
   188:         self.max_grad_norm = max_grad_norm
   189:         self.noise_multiplier = noise_multiplier
   190:         self.n_params = n_params
   191:         self.dataset_size = dataset_size
   192:         self.batch_size = batch_size
   193:         self.epochs = epochs
   194:         self.target_epsilon = target_epsilon
   195:         self.target_delta = target_delta
   196: 
   197:     def clip(self, per_sample_grads, step, epoch):
   198:         """Choose per-sample clipping multipliers.
   199: 
   200:         Args:
   201:             per_sample_grads: list of tensors, each [B, *param_shape] (a copy)
   202:             step: current global training step
   203:             epoch: current epoch number
   204: 
   205:         Returns:
   206:             (scale, clip_norm): scale is a [B] tensor of per-sample multipliers
   207:             (or a list of [B] tensors, one per parameter); clip_norm is the L2
   208:             bound of every scaled per-sample gradient.
   209:         """
   210:         batch_size = per_sample_grads[0].shape[0]
   211: 
   212:         # Compute per-sample gradient norms (flat norm across all parameters)
   213:         flat = torch.cat([g.reshape(batch_size, -1) for g in per_sample_grads], dim=1)
   214:         norms = flat.norm(2, dim=1)  # [B]
   215: 
   216:         # Clip per-sample gradients to max_grad_norm
   217:         clip_factor = (self.max_grad_norm / norms.clamp(min=1e-8)).clamp(max=1.0)  # [B]
   218: 
   219:         return clip_factor, self.max_grad_norm
   220: 
   221:     def get_noise_multiplier(self, step, epoch):
   222:         """Return the noise multiplier sigma_t used (and accounted) this step."""
   223:         return self.noise_multiplier
   224: 
   225:     # Design space: per-sample / per-layer clipping or normalization,
   226:     # adaptive thresholds (C_t), and noise schedules (sigma_t). Helper
   227:     # methods and module-level helpers may live in this section too.
   228:     # The noise itself and the epsilon accounting are done by the harness.
   229: 
   230: 
   231: # =====================================================================
   232: # EDITABLE SECTION END
   233: # =====================================================================
   234: 
   235: 
   236: # =====================================================================
   237: # FIXED: DP privatization and privacy accounting (DO NOT MODIFY)
   238: # =====================================================================
   239: # The Gaussian mechanism and the accountant live here, outside the editable
   240: # section, so the reported epsilon is always computed from the clipping bound
   241: # and noise multiplier that were actually applied. Modules are re-bound under
   242: # private names so that names reassigned in the editable section cannot
   243: # change the mechanism.
   244: import math as _math
   245: import torch as _torch
   246: 
   247: _CLIP_NORM_RTOL = 1e-4  # float slack on the per-sample L2 bound check
   248: _EPSILON_RTOL = 1e-2    # slack for the sigma calibration's binary-search tolerance
   249: 
   250: 
   251: def _rdp_epsilon(steps, sigma, q, delta):
   252:     """Same RDP bound and conversion as compute_epsilon (kept private here)."""
   253:     alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
   254:     best_eps = float("inf")
   255:     for alpha in alphas:
   256:         if alpha <= 1:
   257:             continue
   258:         rdp = steps * min(
   259:             q * q * alpha / (2 * sigma * sigma),
   260:             alpha * q * q / (2 * sigma * sigma),
   261:         )
   262:         eps = rdp - _math.log(delta) / (alpha - 1) + _math.log(1 - 1 / alpha)
   263:         if eps < best_eps:
   264:             best_eps = eps
   265:     return max(0, best_eps)
   266: 
   267: 
   268: class _PrivacyLedger:
   269:     """Composes the per-step noise multipliers the harness actually applied.
   270: 
   271:     Under this RDP bound a step with multiplier sigma_t costs
   272:     alpha * q^2 / (2 sigma_t^2), so T steps compose to the cost of T steps at
   273:     sigma_eff = sqrt(T / sum_t 1/sigma_t^2).
   274:     """
   275: 
   276:     def __init__(self, q, delta):
   277:         self.q = float(q)
   278:         self.delta = float(delta)
   279:         self.steps = 0
   280:         self.inv_sq_sum = 0.0
   281: 
   282:     def record(self, sigma):
   283:         self.steps += 1
   284:         self.inv_sq_sum += 1.0 / (sigma * sigma)
   285: 
   286:     def effective_sigma(self):
   287:         return (self.steps / self.inv_sq_sum) ** 0.5
   288: 
   289:     def epsilon(self):
   290:         return _rdp_epsilon(self.steps, self.effective_sigma(), self.q, self.delta)
   291: 
   292: 
   293: def _positive_finite(value, name):
   294:     value = float(value)
   295:     if not _math.isfinite(value) or value <= 0:
   296:         raise RuntimeError(f"DPMechanism returned invalid {name}={value!r}; "
   297:                            f"it must be a finite positive number")
   298:     return value
   299: 
   300: 
   301: def privatize_step(dp_mechanism, per_sample_grads, step, epoch, ledger):
   302:     """One step of the Gaussian mechanism with the mechanism's C_t and sigma_t.
   303: 
   304:     The mechanism sees a copy of the per-sample gradients and returns
   305:     per-sample multipliers plus the bound C_t; the harness scales the
   306:     original gradients, verifies ||scaled_i|| <= C_t for every sample, averages
   307:     over the batch, adds N(0, (sigma_t * C_t / B)^2) noise and records sigma_t.
   308:     """
   309:     batch_size = per_sample_grads[0].shape[0]
   310:     scale, clip_norm = dp_mechanism.clip(
   311:         [g.clone() for g in per_sample_grads], step, epoch
   312:     )
   313:     clip_norm = _positive_finite(clip_norm, "clip_norm")
   314:     sigma = _positive_finite(dp_mechanism.get_noise_multiplier(step, epoch),
   315:                              "noise multiplier")
   316: 
   317:     scales = list(scale) if isinstance(scale, (list, tuple)) else [scale] * len(per_sample_grads)
   318:     if len(scales) != len(per_sample_grads):
   319:         raise RuntimeError(f"DPMechanism.clip returned {len(scales)} scale tensors "
   320:                            f"for {len(per_sample_grads)} parameters")
   321:     clipped = []
   322:     for g, s in zip(per_sample_grads, scales):
   323:         if type(s) is not _torch.Tensor or tuple(s.shape) != (batch_size,):
   324:             raise RuntimeError("DPMechanism.clip must return per-sample scales as "
   325:                                f"plain torch.Tensor of shape [{batch_size}]")
   326:         shape = [batch_size] + [1] * (g.dim() - 1)
   327:         clipped.append(g * s.detach().reshape(shape))
   328: 
   329:     norms = _torch.cat([c.reshape(batch_size, -1) for c in clipped], dim=1).norm(2, dim=1)
   330:     if not bool((norms <= clip_norm * (1 + _CLIP_NORM_RTOL)).all()):
   331:         raise RuntimeError(
   332:             f"DP violation at step {step}: a scaled per-sample gradient has norm "
   333:             f"{norms.max().item():.6g} > clip_norm={clip_norm:.6g}"
   334:         )
   335: 
   336:     noised_grads = []
   337:     for c in clipped:
   338:         avg = c.mean(dim=0)
   339:         noise = _torch.randn_like(avg) * (sigma * clip_norm / batch_size)
   340:         noised_grads.append(avg + noise)
   341:     ledger.record(sigma)
   342:     return noised_grads
   343: 
   344: 
   345: # =====================================================================
   346: # FIXED: Data loading (DO NOT MODIFY)
   347: # =====================================================================
   348: 
   349: def get_data_loaders(dataset_name, batch_size, data_root=os.environ.get("DATA_ROOT", "/data")):
   350:     """Create train and test data loaders."""
   351:     if dataset_name == "mnist":
   352:         transform = transforms.Compose([
   353:             transforms.ToTensor(),
   354:             transforms.Normalize((0.1307,), (0.3081,)),
   355:         ])
   356:         train_ds = datasets.MNIST(
   357:             os.path.join(data_root, "mnist"), train=True, download=False, transform=transform
   358:         )
   359:         test_ds = datasets.MNIST(
   360:             os.path.join(data_root, "mnist"), train=False, download=False, transform=transform
   361:         )
   362:         model_cls = MNISTNet
   363:     elif dataset_name == "cifar10":
   364:         transform_train = transforms.Compose([
   365:             transforms.ToTensor(),
   366:             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
   367:         ])
   368:         transform_test = transforms.Compose([
   369:             transforms.ToTensor(),
   370:             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
   371:         ])
   372:         train_ds = datasets.CIFAR10(
   373:             os.path.join(data_root, "cifar10"), train=True, download=False, transform=transform_train
   374:         )
   375:         test_ds = datasets.CIFAR10(
   376:             os.path.join(data_root, "cifar10"), train=False, download=False, transform=transform_test
   377:         )
   378:         model_cls = CIFAR10Net
   379:     elif dataset_name == "fmnist":
   380:         transform = transforms.Compose([
   381:             transforms.ToTensor(),
   382:             transforms.Normalize((0.2860,), (0.3530,)),
   383:         ])
   384:         train_ds = datasets.FashionMNIST(
   385:             os.path.join(data_root, "fmnist"), train=True, download=False, transform=transform
   386:         )
   387:         test_ds = datasets.FashionMNIST(
   388:             os.path.join(data_root, "fmnist"), train=False, download=False, transform=transform
   389:         )
   390:         model_cls = MNISTNet
   391:     else:
   392:         raise ValueError(f"Unknown dataset: {dataset_name}")
   393: 
   394:     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
   395:                               num_workers=2, pin_memory=True, drop_last=True)
   396:     test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False,
   397:                              num_workers=2, pin_memory=True)
   398:     return train_ds, train_loader, test_loader, model_cls
   399: 
   400: 
   401: # =====================================================================
   402: # FIXED: Per-sample gradient computation (DO NOT MODIFY)
   403: # =====================================================================
   404: 
   405: def compute_per_sample_gradients(model, data, target, criterion):
   406:     """Compute per-sample gradients using functorch-style vmap.
   407: 
   408:     Returns a list of tensors, each of shape [B, *param_shape].
   409:     """
   410:     params = [p for p in model.parameters() if p.requires_grad]
   411: 
   412:     # Manual per-sample gradient computation via backward on each sample
   413:     batch_size = data.shape[0]
   414:     per_sample_grads = [torch.zeros(batch_size, *p.shape, device=p.device) for p in params]
   415: 
   416:     for i in range(batch_size):
   417:         model.zero_grad()
   418:         output = model(data[i:i+1])
   419:         loss = criterion(output, target[i:i+1])
   420:         loss.backward()
   421:         for j, p in enumerate(params):
   422:             if p.grad is not None:
   423:                 per_sample_grads[j][i] = p.grad.clone()
   424: 
   425:     return per_sample_grads
   426: 
   427: 
   428: def compute_per_sample_gradients_fast(model, data, target, criterion):
   429:     """Efficient per-sample gradient computation using ghost clipping trick.
   430: 
   431:     Computes per-sample gradient norms first, then uses weighted loss for aggregation.
   432:     Falls back to loop-based computation for small batches.
   433:     """
   434:     batch_size = data.shape[0]
   435: 
   436:     # For moderate batch sizes, use vectorized approach via autograd
   437:     if batch_size <= 128:
   438:         return compute_per_sample_gradients(model, data, target, criterion)
   439: 
   440:     # For larger batches, use microbatching for memory efficiency
   441:     micro_bs = 64
   442:     params = [p for p in model.parameters() if p.requires_grad]
   443:     per_sample_grads = [torch.zeros(batch_size, *p.shape, device=p.device) for p in params]
   444: 
   445:     for start in range(0, batch_size, micro_bs):
   446:         end = min(start + micro_bs, batch_size)
   447:         micro_data = data[start:end]
   448:         micro_target = target[start:end]
   449:         for i in range(end - start):
   450:             model.zero_grad()
   451:             output = model(micro_data[i:i+1])
   452:             loss = criterion(output, micro_target[i:i+1])
   453:             loss.backward()
   454:             for j, p in enumerate(params):
   455:                 if p.grad is not None:
   456:                     per_sample_grads[j][start + i] = p.grad.clone()
   457: 
   458:     return per_sample_grads
   459: 
   460: 
   461: # =====================================================================
   462: # FIXED: Training and evaluation loops (DO NOT MODIFY)
   463: # =====================================================================
   464: 
   465: def train_epoch(model, train_loader, optimizer, criterion, dp_mechanism, device,
   466:                 epoch, total_steps, ledger, log_interval=50):
   467:     """Train one epoch with DP mechanism."""
   468:     model.train()
   469:     running_loss = 0.0
   470:     correct = 0
   471:     total = 0
   472:     step = total_steps
   473: 
   474:     for batch_idx, (data, target) in enumerate(train_loader):
   475:         data, target = data.to(device), target.to(device)
   476:         batch_size = data.shape[0]
   477: 
   478:         # Compute per-sample gradients
   479:         per_sample_grads = compute_per_sample_gradients(model, data, target, criterion)
   480: 
   481:         # Apply the DP mechanism: clipping/schedule from DPMechanism (EDITABLE),
   482:         # noise and accounting from privatize_step (FIXED)
   483:         noised_grads = privatize_step(dp_mechanism, per_sample_grads, step, epoch, ledger)
   484: 
   485:         # Set model gradients
   486:         optimizer.zero_grad()
   487:         for param, grad in zip(
   488:             [p for p in model.parameters() if p.requires_grad], noised_grads
   489:         ):
   490:             param.grad = grad
   491: 
   492:         optimizer.step()
   493: 
   494:         # Compute batch metrics (without grad)
   495:         with torch.no_grad():
   496:             output = model(data)
   497:             loss = criterion(output, target)
   498:             running_loss += loss.item() * batch_size
   499:             pred = output.argmax(dim=1)
   500:             correct += pred.eq(target).sum().item()

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
Lines 152–183:
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
   169:     def clip(self, per_sample_grads, step, epoch):
   170:         batch_size = per_sample_grads[0].shape[0]
   171: 
   172:         # Compute per-sample gradient norms (flat norm across all parameters)
   173:         flat = torch.cat([g.reshape(batch_size, -1) for g in per_sample_grads], dim=1)
   174:         norms = flat.norm(2, dim=1)  # [B]
   175: 
   176:         # Clip per-sample gradients to the fixed threshold C
   177:         clip_factor = (self.max_grad_norm / norms.clamp(min=1e-8)).clamp(max=1.0)  # [B]
   178: 
   179:         # The harness adds noise of std sigma * C / B
   180:         return clip_factor, self.max_grad_norm
   181: 
   182:     def get_noise_multiplier(self, step, epoch):
   183:         return self.noise_multiplier
   184: 
   185: 
   186: # =====================================================================
```

### `automatic_clipping` baseline — editable region  [READ-ONLY — reference implementation]

In `opacus/custom_dpsgd.py`:

```python
Lines 152–188:
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
   173:     def clip(self, per_sample_grads, step, epoch):
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
   184:         # Sensitivity bound C=1 for AUTO-S: the harness adds noise sigma * 1 / B
   185:         return scale, 1.0
   186: 
   187:     def get_noise_multiplier(self, step, epoch):
   188:         return self.noise_multiplier
   189: 
   190: 
   191: # =====================================================================
```

### `adaptive_clipping` baseline — editable region  [READ-ONLY — reference implementation]

In `opacus/custom_dpsgd.py`:

```python
Lines 152–200:
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
   176:     def clip(self, per_sample_grads, step, epoch):
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
   196:         # The harness adds noise calibrated to the current clip norm
   197:         return clip_factor, self.clip_norm
   198: 
   199:     def get_noise_multiplier(self, step, epoch):
   200:         return self.noise_multiplier
   201: 
   202: 
   203: # =====================================================================
```

### `noise_decay` baseline — editable region  [READ-ONLY — reference implementation]

In `opacus/custom_dpsgd.py`:

```python
Lines 152–230:
   149:     return (sigma_low + sigma_high) / 2
   150: 
   151: 
   152: class DPMechanism:
   153:     """Step-Decay Noise Schedule (inspired by Global-Adapt-V2-S, 2025).
   154: 
   155:     Decays noise multiplier and clipping threshold over training epochs
   156:     to allocate more privacy budget to later (more useful) training steps.
   157: 
   158:     Privacy accounting: sigma_0 is chosen so that the full schedule spends
   159:     the same budget as the calibrated uniform sigma; the fixed harness
   160:     composes the per-step sigma it actually applied.
   161:     """
   162: 
   163:     def __init__(self, max_grad_norm, noise_multiplier, n_params,
   164:                  dataset_size, batch_size, epochs, target_epsilon, target_delta):
   165:         self.max_grad_norm = max_grad_norm
   166:         self.noise_multiplier = noise_multiplier
   167:         self.n_params = n_params
   168:         self.dataset_size = dataset_size
   169:         self.batch_size = batch_size
   170:         self.epochs = epochs
   171:         self.target_epsilon = target_epsilon
   172:         self.target_delta = target_delta
   173: 
   174:         # Step-decay schedule parameters
   175:         # Decay noise and clipping every decay_interval epochs
   176:         self.decay_interval = max(1, epochs // 4)  # 4 decay stages
   177:         self.noise_decay_factor = 0.8  # Reduce noise by 20% at each stage
   178:         self.clip_decay_factor = 0.85  # Reduce clip norm by 15% at each stage
   179: 
   180:         # Pre-compute the per-epoch sigma schedule so we can do accurate
   181:         # RDP accounting.  Steps per epoch = dataset_size // batch_size
   182:         # (drop_last=True in DataLoader).
   183:         self.steps_per_epoch = dataset_size // batch_size
   184: 
   185:         # Compute sigma_0: scale the calibrated (uniform) sigma up so that
   186:         # the harmonic-mean-equivalent sigma across all steps equals the
   187:         # calibrated value.  This keeps the total privacy spend equal to
   188:         # the budget even though individual steps have different noise.
   189:         total_steps = self.steps_per_epoch * epochs
   190:         inv_sq_sum = 0.0
   191:         for e in range(1, epochs + 1):
   192:             stage = (e - 1) // self.decay_interval
   193:             factor = self.noise_decay_factor ** stage
   194:             # Each epoch contributes steps_per_epoch steps at sigma_0*factor
   195:             # 1/sigma_t^2 = 1/(sigma_0*factor)^2 = 1/(sigma_0^2 * factor^2)
   196:             inv_sq_sum += self.steps_per_epoch / (factor * factor)
   197:         # sigma_eff = sqrt(total_steps / inv_sq_sum) * sigma_0
   198:         # We want sigma_eff == noise_multiplier (the calibrated value), so:
   199:         #   noise_multiplier = sigma_0 * sqrt(total_steps / inv_sq_sum)
   200:         #   sigma_0 = noise_multiplier / sqrt(total_steps / inv_sq_sum)
   201:         #           = noise_multiplier * sqrt(inv_sq_sum / total_steps)
   202:         self.sigma_0 = noise_multiplier * (inv_sq_sum / total_steps) ** 0.5
   203:         self.clip_0 = max_grad_norm
   204: 
   205:         # Current values
   206:         self._current_sigma = self.sigma_0
   207:         self._current_clip = self.clip_0
   208: 
   209:     def clip(self, per_sample_grads, step, epoch):
   210:         batch_size = per_sample_grads[0].shape[0]
   211: 
   212:         # Update schedule based on epoch
   213:         stage = (epoch - 1) // self.decay_interval
   214:         self._current_sigma = self.sigma_0 * (self.noise_decay_factor ** stage)
   215:         self._current_clip = self.clip_0 * (self.clip_decay_factor ** stage)
   216: 
   217:         # Compute per-sample gradient norms
   218:         flat = torch.cat([g.reshape(batch_size, -1) for g in per_sample_grads], dim=1)
   219:         norms = flat.norm(2, dim=1)  # [B]
   220: 
   221:         # Clip per-sample gradients using current (decayed) threshold
   222:         clip_factor = (self._current_clip / norms.clamp(min=1e-8)).clamp(max=1.0)
   223: 
   224:         # The harness adds noise calibrated to the current clip norm and sigma
   225:         return clip_factor, self._current_clip
   226: 
   227:     def get_noise_multiplier(self, step, epoch):
   228:         """Current (decayed) noise multiplier; the harness accounts each
   229:         step with the sigma it actually applied."""
   230:         return self._current_sigma
   231: 
   232: 
   233: # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

## Time Budget

You have **5 hours** of wall-clock time before submission, covering
everything you do here: reading the code, editing it, and any trial runs
you launch.

Good luck.
