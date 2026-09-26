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
    80: # These helpers use the harness's accountant (_rdp_epsilon, in the FIXED
    81: # section after DPMechanism; call them at run time): subsampled-Gaussian RDP
    82: # (Mironov et al. 2019, as in Opacus) composed over steps, then (eps, delta).
    83: 
    84: def compute_epsilon(steps, sigma, q, delta, alphas=None):
    85:     """Compute (epsilon, best_alpha) via RDP accounting.
    86: 
    87:     Args:
    88:         steps: number of training steps
    89:         sigma: noise multiplier
    90:         q: sampling probability (batch_size / dataset_size)
    91:         delta: target delta
    92:         alphas: list of RDP orders to try
    93: 
    94:     Returns:
    95:         (epsilon, best_alpha)
    96:     """
    97:     return compute_epsilon_schedule([(sigma, steps)], q, delta, alphas)
    98: 
    99: 
   100: def compute_epsilon_schedule(schedule, q, delta, alphas=None):
   101:     """(epsilon, best_alpha) of a noise schedule given as (sigma, n_steps)
   102:     pairs, composed exactly as the harness's privacy ledger does."""
   103:     return _rdp_epsilon(schedule, q, delta, alphas)
   104: 
   105: 
   106: def calibrate_noise_to_epsilon(target_epsilon, steps, q, delta, tol=1e-3):
   107:     """Find the noise multiplier sigma that achieves target_epsilon.
   108: 
   109:     Uses binary search to find the right noise level.
   110:     """
   111:     sigma_low, sigma_high = 0.01, 100.0
   112:     while sigma_high - sigma_low > tol:
   113:         sigma_mid = (sigma_low + sigma_high) / 2
   114:         eps, _ = compute_epsilon(steps, sigma_mid, q, delta)
   115:         if eps > target_epsilon:
   116:             sigma_low = sigma_mid
   117:         else:
   118:             sigma_high = sigma_mid
   119:     return (sigma_low + sigma_high) / 2
   120: 
   121: 
   122: # Originals of what the FIXED harness relies on (noise, clipping check,
   123: # accountant, accuracy count), captured before the editable section runs.
   124: # ("module.<builtin>" must stay absent: fixed code looks these names up.)
   125: import builtins as _builtins
   126: from scipy import special as _sp_special
   127: _HARNESS_BUILTINS = ("bool", "float", "int", "type", "len", "isinstance", "list",
   128:                      "tuple", "zip", "range", "enumerate", "min", "max", "abs",
   129:                      "print", "RuntimeError")
   130: _HARNESS_REFS = tuple(
   131:     (f"{label}.{name}", ns, name, ns.get(name))
   132:     for label, ns, names in (
   133:         ("torch", vars(torch), ("Tensor", "randn_like", "cat")),
   134:         ("torch.linalg", vars(torch.linalg), ("vector_norm",)),
   135:         ("torch._C", vars(torch._C), ("_len_torch_function_stack",
   136:                                       "_len_torch_dispatch_stack")),
   137:         ("torch.Tensor", vars(torch.Tensor), (
   138:             "__torch_function__", "__getattribute__", "__getattr__", "shape",
   139:             "to", "clone", "detach", "reshape", "dim", "__mul__", "__add__",
   140:             "__le__", "__bool__", "all", "mean", "max", "item", "argmax", "eq",
   141:             "sum")),
   142:         ("math", vars(math), ("log", "log1p", "exp", "expm1", "sqrt", "isfinite")),
   143:         ("scipy.special", vars(_sp_special), ("binom", "log_ndtr")),
   144:         ("builtins", vars(_builtins), _HARNESS_BUILTINS),
   145:         ("module", globals(), ("torch", "print", "range", "enumerate", "zip",
   146:                                "len", "RuntimeError")),
   147:     )
   148:     for name in names
   149: )
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
   241: # and noise multiplier that were actually applied.
   242: #
   243: # What is protected against code in the editable section:
   244: # - The fixed functions below take the helpers and library functions they use
   245: #   (the originals in _HARNESS_REFS) as default arguments or closure variables
   246: #   bound at definition time, so reassigning a module-level name (e.g. with
   247: #   `global`) or a torch/math attribute does not change them.
   248: # - _check_harness() runs after every call into DPMechanism, before the
   249: #   harness uses its result, and before the final metrics are printed. It
   250: #   aborts the run if an entry of _HARNESS_REFS or a fixed name of this file
   251: #   defined below was replaced (or a listed builtin shadowed), or if a torch
   252: #   function/dispatch mode is active.
   253: # Not covered (in-process residual): code that deliberately rewrites the
   254: # harness's own objects (_HARNESS_REFS before it is bound, function defaults
   255: # or closures, frames, gc) or forges the printed output.
   256: _ORIG = {key: obj for key, ns, name, obj in _HARNESS_REFS}
   257: 
   258: _CLIP_NORM_RTOL = 1e-4  # float slack on the per-sample L2 bound check
   259: _EPSILON_RTOL = 1e-2    # slack for the sigma calibration's binary-search tolerance
   260: _RDP_ORDERS = tuple([1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)))
   261: 
   262: 
   263: def _make_harness_check(refs=_HARNESS_REFS, _RuntimeError=_ORIG["builtins.RuntimeError"]):
   264:     fixed = []  # (key, namespace, name, object) of the fixed code, see seal()
   265:     mode_stacks = [obj for key, ns, name, obj in refs if key.startswith("torch._C.")]
   266: 
   267:     def check():
   268:         for entries in (refs, fixed):
   269:             for key, ns, name, obj in entries:
   270:                 if ns.get(name) is not obj:
   271:                     raise _RuntimeError(
   272:                         f"DP harness tampering: `{key}` was replaced or shadowed; "
   273:                         f"the fixed harness relies on the original")
   274:         for stack_len in mode_stacks:
   275:             if stack_len():
   276:                 raise _RuntimeError("DP harness tampering: a torch function or "
   277:                                     "dispatch mode is active")
   278: 
   279:     def seal(*groups):
   280:         if fixed:
   281:             raise _RuntimeError("the DP harness is already sealed")
   282:         for label, ns, names in groups:
   283:             for name in names:
   284:                 fixed.append((f"{label}.{name}", ns, name, ns[name]))
   285:         check()
   286: 
   287:     return check, seal
   288: 
   289: 
   290: _check_harness, _seal_harness = _make_harness_check()
   291: _check_harness()  # module-level code of the editable section replaced nothing
   292: 
   293: 
   294: def _make_rdp_accountant(
   295:         orders=_RDP_ORDERS, log=_ORIG["math.log"], log1p=_ORIG["math.log1p"],
   296:         exp=_ORIG["math.exp"], expm1=_ORIG["math.expm1"], sqrt=_ORIG["math.sqrt"],
   297:         binom=_ORIG["scipy.special.binom"], log_ndtr=_ORIG["scipy.special.log_ndtr"],
   298:         _float=_ORIG["builtins.float"], _int=_ORIG["builtins.int"],
   299:         _abs=_ORIG["builtins.abs"], _min=_ORIG["builtins.min"],
   300:         _max=_ORIG["builtins.max"], _range=_ORIG["builtins.range"],
   301:         _zip=_ORIG["builtins.zip"], _len=_ORIG["builtins.len"],
   302:         _tuple=_ORIG["builtins.tuple"]):
   303:     """RDP of the subsampled Gaussian mechanism with sampling rate q (Mironov,
   304:     Talwar and Zhang 2019, "Renyi Differential Privacy of the Sampled Gaussian
   305:     Mechanism"; the computation used by Opacus and TF-Privacy), composed per
   306:     order over a noise schedule and converted to (epsilon, delta) with
   307:     eps = rdp + log(1/delta)/(alpha-1) + log(1-1/alpha), minimised over alpha.
   308:     """
   309:     inf = _float("inf")
   310: 
   311:     def log_add(x, y):
   312:         a, b = _min(x, y), _max(x, y)
   313:         return b if a == -inf else log1p(exp(a - b)) + b
   314: 
   315:     def log_sub(x, y):
   316:         if x < y:
   317:             raise ValueError("The result of subtraction must be non-negative.")
   318:         if y == -inf:
   319:             return x
   320:         if x == y:
   321:             return -inf
   322:         try:
   323:             return log(expm1(x - y)) + y
   324:         except OverflowError:
   325:             return x
   326: 
   327:     def log_erfc(x):
   328:         return log(2) + log_ndtr(-x * 2 ** 0.5)
   329: 
   330:     def log_a_int(q, sigma, alpha):
   331:         log_a = -inf
   332:         for i in _range(alpha + 1):
   333:             log_coef_i = log(binom(alpha, i)) + i * log(q) + (alpha - i) * log(1 - q)
   334:             log_a = log_add(log_a, log_coef_i + (i * i - i) / (2 * (sigma ** 2)))
   335:         return _float(log_a)
   336: 
   337:     def log_a_frac(q, sigma, alpha):
   338:         log_a0, log_a1 = -inf, -inf
   339:         i = 0
   340:         z0 = sigma ** 2 * log(1 / q - 1) + 0.5
   341:         while True:
   342:             coef = binom(alpha, i)
   343:             log_coef = log(_abs(coef))
   344:             j = alpha - i
   345:             log_t0 = log_coef + i * log(q) + j * log(1 - q)
   346:             log_t1 = log_coef + j * log(q) + i * log(1 - q)
   347:             log_e0 = log(0.5) + log_erfc((i - z0) / (sqrt(2) * sigma))
   348:             log_e1 = log(0.5) + log_erfc((z0 - j) / (sqrt(2) * sigma))
   349:             log_s0 = log_t0 + (i * i - i) / (2 * (sigma ** 2)) + log_e0
   350:             log_s1 = log_t1 + (j * j - j) / (2 * (sigma ** 2)) + log_e1
   351:             if coef > 0:
   352:                 log_a0, log_a1 = log_add(log_a0, log_s0), log_add(log_a1, log_s1)
   353:             else:
   354:                 log_a0, log_a1 = log_sub(log_a0, log_s0), log_sub(log_a1, log_s1)
   355:             i += 1
   356:             if _max(log_s0, log_s1) < -30:
   357:                 break
   358:         return log_add(log_a0, log_a1)
   359: 
   360:     def step_rdp(q, sigma, alpha):
   361:         if q == 0:
   362:             return 0.0
   363:         if sigma == 0:
   364:             return inf
   365:         if q == 1.0:
   366:             return alpha / (2 * sigma ** 2)
   367:         if _float(alpha).is_integer():
   368:             return log_a_int(q, sigma, _int(alpha)) / (alpha - 1)
   369:         return log_a_frac(q, sigma, alpha) / (alpha - 1)
   370: 
   371:     def epsilon(schedule, q, delta, alphas=None, cache=None):
   372:         """(epsilon, best_alpha) of `schedule`, a list of (sigma, n_steps);
   373:         `cache` (a dict) keeps each sigma's one-step RDP across calls."""
   374:         q, delta = _float(q), _float(delta)
   375:         alphas = orders if alphas is None else _tuple(alphas)
   376:         cache = {} if cache is None else cache
   377:         rdp = [0.0] * _len(alphas)
   378:         for sigma, n_steps in schedule:
   379:             key = (q, _float(sigma), alphas)
   380:             per_step = cache.get(key)
   381:             if per_step is None:
   382:                 per_step = cache[key] = [step_rdp(q, key[1], a) for a in alphas]
   383:             rdp = [r + n_steps * s for r, s in _zip(rdp, per_step)]
   384:         best_eps, best_alpha = inf, None
   385:         for alpha, r in _zip(alphas, rdp):
   386:             if alpha <= 1:
   387:                 continue
   388:             eps = r - log(delta) / (alpha - 1) + log(1 - 1 / alpha)
   389:             if eps < best_eps:
   390:                 best_eps, best_alpha = eps, alpha
   391:         return _max(0, best_eps), best_alpha
   392: 
   393:     return epsilon
   394: 
   395: 
   396: _rdp_epsilon = _make_rdp_accountant()
   397: 
   398: 
   399: class _PrivacyLedger:
   400:     """Counts the steps run at each noise multiplier sigma_t the harness
   401:     applied; epsilon() composes their subsampled-Gaussian RDP per order (no
   402:     shortcut through an "effective" sigma)."""
   403: 
   404:     def __init__(self, q, delta, _float=_ORIG["builtins.float"]):
   405:         self.q = _float(q)
   406:         self.delta = _float(delta)
   407:         self.steps = 0
   408:         self.last_sigma = None
   409:         self.counts = {}  # sigma_t -> number of steps
   410:         self.rdp_cache = {}
   411: 
   412:     def record(self, sigma):
   413:         self.steps += 1
   414:         self.last_sigma = sigma
   415:         self.counts[sigma] = self.counts.get(sigma, 0) + 1
   416: 
   417:     def epsilon(self, _epsilon=_rdp_epsilon, _list=_ORIG["builtins.list"]):
   418:         return _epsilon(_list(self.counts.items()), self.q, self.delta,
   419:                         cache=self.rdp_cache)[0]
   420: 
   421: 
   422: def _positive_finite(value, name, _float=_ORIG["builtins.float"],
   423:                      _isfinite=_ORIG["math.isfinite"]):
   424:     value = _float(value)
   425:     if not _isfinite(value) or value <= 0:
   426:         raise RuntimeError(f"DPMechanism returned invalid {name}={value!r}; "
   427:                            f"it must be a finite positive number")
   428:     return value
   429: 
   430: 
   431: def privatize_step(dp_mechanism, per_sample_grads, step, epoch, ledger,
   432:                    _check=_check_harness, _positive=_positive_finite,
   433:                    _Tensor=_ORIG["torch.Tensor"], _randn_like=_ORIG["torch.randn_like"],
   434:                    _cat=_ORIG["torch.cat"], _vector_norm=_ORIG["torch.linalg.vector_norm"],
   435:                    _rtol=_CLIP_NORM_RTOL, _bool=_ORIG["builtins.bool"],
   436:                    _type=_ORIG["builtins.type"], _len=_ORIG["builtins.len"],
   437:                    _isinstance=_ORIG["builtins.isinstance"], _list=_ORIG["builtins.list"],
   438:                    _tuple=_ORIG["builtins.tuple"], _zip=_ORIG["builtins.zip"]):
   439:     """One step of the Gaussian mechanism with the mechanism's C_t and sigma_t.
   440: 
   441:     The mechanism sees a copy of the per-sample gradients and returns
   442:     per-sample multipliers plus the bound C_t; the harness scales the
   443:     original gradients, verifies ||scaled_i|| <= C_t for every sample, averages
   444:     over the batch, adds N(0, (sigma_t * C_t / B)^2) noise and records sigma_t.
   445:     """
   446:     batch_size = per_sample_grads[0].shape[0]
   447:     scale, clip_norm = dp_mechanism.clip(
   448:         [g.clone() for g in per_sample_grads], step, epoch
   449:     )
   450:     clip_norm = _positive(clip_norm, "clip_norm")
   451:     sigma = _positive(dp_mechanism.get_noise_multiplier(step, epoch),
   452:                       "noise multiplier")
   453:     scales = _list(scale) if _isinstance(scale, (_list, _tuple)) else [scale] * _len(per_sample_grads)
   454:     _check()  # no DPMechanism code runs past this point in this step
   455: 
   456:     if _len(scales) != _len(per_sample_grads):
   457:         raise RuntimeError(f"DPMechanism.clip returned {_len(scales)} scale tensors "
   458:                            f"for {_len(per_sample_grads)} parameters")
   459:     clipped = []
   460:     for g, s in _zip(per_sample_grads, scales):
   461:         if _type(s) is not _Tensor or _tuple(s.shape) != (batch_size,):
   462:             raise RuntimeError("DPMechanism.clip must return per-sample scales as "
   463:                                f"plain torch.Tensor of shape [{batch_size}]")
   464:         shape = [batch_size] + [1] * (g.dim() - 1)
   465:         clipped.append(g * s.detach().to(device=g.device, dtype=g.dtype).reshape(shape))
   466: 
   467:     norms = _vector_norm(_cat([c.reshape(batch_size, -1) for c in clipped], dim=1), 2, dim=1)
   468:     if not _bool((norms <= clip_norm * (1 + _rtol)).all()):
   469:         raise RuntimeError(
   470:             f"DP violation at step {step}: a scaled per-sample gradient has norm "
   471:             f"{norms.max().item():.6g} > clip_norm={clip_norm:.6g}"
   472:         )
   473: 
   474:     noised_grads = []
   475:     for c in clipped:
   476:         avg = c.mean(dim=0)
   477:         noise = _randn_like(avg) * (sigma * clip_norm / batch_size)
   478:         noised_grads.append(avg + noise)
   479:     ledger.record(sigma)
   480:     return noised_grads
   481: 
   482: 
   483: # =====================================================================
   484: # FIXED: Data loading (DO NOT MODIFY)
   485: # =====================================================================
   486: 
   487: def get_data_loaders(dataset_name, batch_size, data_root=os.environ.get("DATA_ROOT", "/data")):
   488:     """Create train and test data loaders."""
   489:     if dataset_name == "mnist":
   490:         transform = transforms.Compose([
   491:             transforms.ToTensor(),
   492:             transforms.Normalize((0.1307,), (0.3081,)),
   493:         ])
   494:         train_ds = datasets.MNIST(
   495:             os.path.join(data_root, "mnist"), train=True, download=False, transform=transform
   496:         )
   497:         test_ds = datasets.MNIST(
   498:             os.path.join(data_root, "mnist"), train=False, download=False, transform=transform
   499:         )
   500:         model_cls = MNISTNet

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
   149: )
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
   149: )
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
   149: )
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
Lines 152–231:
   149: )
   150: 
   151: 
   152: class DPMechanism:
   153:     """Step-Decay Noise Schedule (inspired by Global-Adapt-V2-S, 2025).
   154: 
   155:     Decays noise multiplier and clipping threshold over training epochs
   156:     to allocate more privacy budget to later (more useful) training steps.
   157: 
   158:     Privacy accounting: sigma_0 is calibrated so that the full decayed
   159:     schedule spends the target budget under the harness's accountant
   160:     (compute_epsilon_schedule); the fixed harness composes the per-step
   161:     sigma it actually applied.
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
   181:         # Per-epoch sigma schedule for the RDP accounting. Steps per epoch =
   182:         # dataset_size // batch_size (drop_last=True in DataLoader).
   183:         self.steps_per_epoch = dataset_size // batch_size
   184:         q = batch_size / dataset_size
   185: 
   186:         def schedule(sigma_0):
   187:             return [(sigma_0 * (self.noise_decay_factor ** ((e - 1) // self.decay_interval)),
   188:                      self.steps_per_epoch) for e in range(1, epochs + 1)]
   189: 
   190:         # Calibrate sigma_0 by bisection: the smallest sigma_0 whose decayed
   191:         # schedule stays within the budget. sigma_0 = noise_multiplier (the
   192:         # calibrated uniform sigma) over-spends once sigma decays; at the upper
   193:         # end every step's sigma is >= noise_multiplier.
   194:         lo = noise_multiplier
   195:         hi = noise_multiplier / self.noise_decay_factor ** ((epochs - 1) // self.decay_interval)
   196:         while hi - lo > 1e-4 * hi:
   197:             mid = (lo + hi) / 2
   198:             eps, _ = compute_epsilon_schedule(schedule(mid), q, target_delta)
   199:             if eps > target_epsilon:
   200:                 lo = mid
   201:             else:
   202:                 hi = mid
   203:         self.sigma_0 = hi
   204:         self.clip_0 = max_grad_norm
   205: 
   206:         # Current values
   207:         self._current_sigma = self.sigma_0
   208:         self._current_clip = self.clip_0
   209: 
   210:     def clip(self, per_sample_grads, step, epoch):
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
   225:         # The harness adds noise calibrated to the current clip norm and sigma
   226:         return clip_factor, self._current_clip
   227: 
   228:     def get_noise_multiplier(self, step, epoch):
   229:         """Current (decayed) noise multiplier; the harness accounts each
   230:         step with the sigma it actually applied."""
   231:         return self._current_sigma
   232: 
   233: 
   234: # =====================================================================
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
