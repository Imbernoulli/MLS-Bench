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
- The total privacy budget `(target_epsilon, target_delta)` is FIXED and checked externally: the harness composes the per-step `σ_t` it applied, and a run whose epsilon exceeds the target by more than 1% (the slack for the noise calibration's binary search), or whose clipped per-sample gradient exceeds its declared `C_t`, is aborted.
- The model architecture, data pipeline, optimizer, and training loop are FIXED; the harness aborts a run that hooks or monkeypatches the functions it relies on (see `_check_harness` in the file).
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
    85:     """(epsilon, best_alpha) after `steps` steps at noise multiplier `sigma`,
    86:     sampling rate q = batch_size / dataset_size, over the RDP orders `alphas`."""
    87:     return compute_epsilon_schedule([(sigma, steps)], q, delta, alphas)
    88: 
    89: 
    90: def compute_epsilon_schedule(schedule, q, delta, alphas=None):
    91:     """(epsilon, best_alpha) of a noise schedule given as (sigma, n_steps)
    92:     pairs, composed exactly as the harness's privacy ledger does."""
    93:     return _rdp_epsilon(schedule, q, delta, alphas)
    94: 
    95: 
    96: def calibrate_noise_to_epsilon(target_epsilon, steps, q, delta, tol=1e-3):
    97:     """Binary-search the noise multiplier sigma that achieves target_epsilon."""
    98:     sigma_low, sigma_high = 0.01, 100.0
    99:     while sigma_high - sigma_low > tol:
   100:         sigma_mid = (sigma_low + sigma_high) / 2
   101:         eps, _ = compute_epsilon(steps, sigma_mid, q, delta)
   102:         if eps > target_epsilon:
   103:             sigma_low = sigma_mid
   104:         else:
   105:             sigma_high = sigma_mid
   106:     return (sigma_low + sigma_high) / 2
   107: 
   108: 
   109: # Originals of what the FIXED harness relies on (noise, clipping check,
   110: # accountant, model calls, parameter update, accuracy count), captured before
   111: # the editable section runs; _check_harness (below) compares against them.
   112: import builtins as _builtins, torch.utils._device as _tdev
   113: import torch.nn.modules.module as _tmm, torch.optim.optimizer as _topt, torch.optim.sgd as _tsgd
   114: from scipy import special as _sp_special
   115: optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)  # wraps SGD.step (profiler) now
   116: _HARNESS_BUILTINS = ("bool", "float", "int", "type", "len", "isinstance", "list", "tuple", "zip",
   117:                      "range", "enumerate", "min", "max", "abs", "vars", "print", "RuntimeError")
   118: _HARNESS_REFS = tuple(
   119:     (f"{label}.{name}", ns, name, ns.get(name))
   120:     for label, ns, names in (
   121:         ("torch", vars(torch), ("Tensor", "randn_like", "cat", "_foreach_add", "_foreach_add_",
   122:                                 "_foreach_mul", "_foreach_mul_", "_foreach_neg")),
   123:         ("torch.linalg", vars(torch.linalg), ("vector_norm",)),
   124:         ("torch._C", vars(torch._C), ("_len_torch_function_stack", "_get_function_stack_at",
   125:                                       "_len_torch_dispatch_stack")),
   126:         ("torch.Tensor", vars(torch.Tensor), (
   127:             "__torch_function__", "__getattribute__", "__getattr__", "shape", "grad", "to",
   128:             "clone", "detach", "reshape", "dim", "__mul__", "__add__", "__le__", "__bool__",
   129:             "all", "mean", "max", "item", "argmax", "eq", "sum")),
   130:         ("torch.nn.Module", vars(nn.Module), ("__call__", "_call_impl", "_wrapped_call_impl",
   131:                                               "__getattr__", "modules", "named_modules")),
   132:         ("torch.nn.modules.module", vars(_tmm), [n for n in vars(_tmm) if n.startswith("_global_")]),
   133:         ("torch.optim.optimizer", vars(_topt), ("_global_optimizer_pre_hooks",
   134:                                                 "_global_optimizer_post_hooks")),
   135:         ("torch.optim", vars(optim), ("SGD",)),
   136:         # (not SGD._init_group / sgd._fused_sgd: torch.compile legitimately rewraps them)
   137:         ("torch.optim.SGD", vars(optim.SGD), ("step",)),
   138:         ("torch.optim.sgd", vars(_tsgd), ("sgd", "_single_tensor_sgd", "_multi_tensor_sgd")),
   139:         ("torch.utils._device", vars(_tdev), ("DeviceContext",)),
   140:         ("DeviceContext", vars(_tdev.DeviceContext), ("__torch_function__",)),
   141:         ("math", vars(math), ("log", "log1p", "exp", "expm1", "sqrt", "isfinite")),
   142:         ("scipy.special", vars(_sp_special), ("binom", "log_ndtr")),
   143:         ("builtins", vars(_builtins), _HARNESS_BUILTINS),
   144:         # fixed module-level names; the builtins must stay unshadowed (absent) here
   145:         ("module", globals(), ("torch", "nn", "optim", "print", "range", "enumerate", "zip",
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
   176: #   whose epsilon exceeds 1.01 x target_epsilon (calibration slack); a scaled
   177: #   per-sample gradient whose norm exceeds clip_norm also aborts the run.
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
   247: #   `global`) or a torch/math attribute does not change them. Methods of the
   248: #   tensors DPMechanism returns are called unbound (never looked up on them).
   249: # - _check_harness() runs after every call into DPMechanism (before the harness
   250: #   uses its result), before every optimizer step, before every evaluation and
   251: #   before the final metrics are printed. It aborts the run if
   252: #     * an entry of _HARNESS_REFS (the listed torch / torch.nn.Module /
   253: #       torch.optim / math / scipy.special / builtins attributes and module
   254: #       names) or a fixed name defined below was replaced, or a listed builtin
   255: #       is shadowed at module level;
   256: #     * a global torch.nn module hook or optimizer step hook is registered, or
   257: #       the harness's model, criterion or optimizer carries its own
   258: #       forward/backward/step hooks or an instance-level `forward`;
   259: #     * a torch function mode other than torch's own DeviceContext (set by
   260: #       torch.set_default_device) or any torch dispatch mode is active.
   261: # Not covered (in-process residual): code that deliberately rewrites the
   262: # harness's own objects (_HARNESS_REFS before it is bound, function defaults
   263: # or closures, frames, gc), torch internals not listed above (e.g. kernel
   264: # overrides registered through torch.library, layer/loss implementations),
   265: # and forged printed output.
   266: _ORIG = {key: obj for key, ns, name, obj in _HARNESS_REFS}
   267: 
   268: _CLIP_NORM_RTOL = 1e-4  # float slack on the per-sample L2 bound check
   269: _EPSILON_RTOL = 1e-2    # slack for the sigma calibration's binary-search tolerance
   270: _RDP_ORDERS = tuple([1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)))
   271: _MODULE_HOOKS = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks",
   272:                  "_backward_pre_hooks")
   273: _OPTIMIZER_HOOKS = ("_optimizer_step_pre_hooks", "_optimizer_step_post_hooks")
   274: 
   275: 
   276: def _make_harness_check(
   277:         refs=_HARNESS_REFS, _RuntimeError=_ORIG["builtins.RuntimeError"],
   278:         _type=_ORIG["builtins.type"], _range=_ORIG["builtins.range"],
   279:         _vars=_ORIG["builtins.vars"], _modules=_ORIG["torch.nn.Module.modules"],
   280:         _tf_len=_ORIG["torch._C._len_torch_function_stack"],
   281:         _tf_at=_ORIG["torch._C._get_function_stack_at"],
   282:         _dispatch_len=_ORIG["torch._C._len_torch_dispatch_stack"],
   283:         _DeviceContext=_ORIG["torch.utils._device.DeviceContext"],
   284:         module_hooks=_MODULE_HOOKS, optimizer_hooks=_OPTIMIZER_HOOKS):
   285:     fixed = []  # (key, namespace, name, object) of the fixed code, see seal()
   286:     registries = [(key, obj) for key, ns, name, obj in refs  # global hook dicts
   287:                   if name.startswith("_global_") and obj is not None]
   288: 
   289:     def tampered(what):
   290:         return _RuntimeError(f"DP harness tampering: {what}")
   291: 
   292:     def check(*modules, optimizer=None):
   293:         for entries in (refs, fixed):
   294:             for key, ns, name, obj in entries:
   295:                 if ns.get(name) is not obj:
   296:                     raise tampered(f"`{key}` was replaced or shadowed; the fixed "
   297:                                    f"harness relies on the original")
   298:         for key, registry in registries:
   299:             if registry:
   300:                 raise tampered(f"a global hook is registered in `{key}`")
   301:         for i in _range(_tf_len()):
   302:             if _type(_tf_at(i)) is not _DeviceContext:
   303:                 raise tampered("a torch function mode is active")
   304:         if _dispatch_len():
   305:             raise tampered("a torch dispatch mode is active")
   306:         for root in modules:
   307:             for module in _modules(root):
   308:                 ns = _vars(module)
   309:                 if "forward" in ns:
   310:                     raise tampered(f"{_type(module).__name__}.forward was overridden")
   311:                 for name in module_hooks:
   312:                     if ns.get(name):
   313:                         raise tampered(f"{_type(module).__name__} has {name}")
   314:         if optimizer is not None:
   315:             ns = _vars(optimizer)
   316:             for name in optimizer_hooks:
   317:                 if ns.get(name):
   318:                     raise tampered(f"the optimizer has {name}")
   319: 
   320:     def seal(*groups):
   321:         if fixed:
   322:             raise _RuntimeError("the DP harness is already sealed")
   323:         for label, ns, names in groups:
   324:             for name in names:
   325:                 fixed.append((f"{label}.{name}", ns, name, ns[name]))
   326:         check()
   327: 
   328:     return check, seal
   329: 
   330: 
   331: _check_harness, _seal_harness = _make_harness_check()
   332: _check_harness()  # module-level code of the editable section replaced nothing
   333: 
   334: 
   335: def _make_rdp_accountant(
   336:         orders=_RDP_ORDERS, log=_ORIG["math.log"], log1p=_ORIG["math.log1p"],
   337:         exp=_ORIG["math.exp"], expm1=_ORIG["math.expm1"], sqrt=_ORIG["math.sqrt"],
   338:         binom=_ORIG["scipy.special.binom"], log_ndtr=_ORIG["scipy.special.log_ndtr"],
   339:         _float=_ORIG["builtins.float"], _int=_ORIG["builtins.int"],
   340:         _abs=_ORIG["builtins.abs"], _min=_ORIG["builtins.min"],
   341:         _max=_ORIG["builtins.max"], _range=_ORIG["builtins.range"],
   342:         _zip=_ORIG["builtins.zip"], _len=_ORIG["builtins.len"],
   343:         _tuple=_ORIG["builtins.tuple"]):
   344:     """RDP of the subsampled Gaussian mechanism with sampling rate q (Mironov,
   345:     Talwar and Zhang 2019, "Renyi Differential Privacy of the Sampled Gaussian
   346:     Mechanism"; the computation used by Opacus and TF-Privacy), composed per
   347:     order over a noise schedule and converted to (epsilon, delta) with
   348:     eps = rdp + log(1/delta)/(alpha-1) + log(1-1/alpha), minimised over alpha.
   349:     """
   350:     inf = _float("inf")
   351: 
   352:     def log_add(x, y):
   353:         a, b = _min(x, y), _max(x, y)
   354:         return b if a == -inf else log1p(exp(a - b)) + b
   355: 
   356:     def log_sub(x, y):
   357:         if x < y:
   358:             raise ValueError("The result of subtraction must be non-negative.")
   359:         if y == -inf:
   360:             return x
   361:         if x == y:
   362:             return -inf
   363:         try:
   364:             return log(expm1(x - y)) + y
   365:         except OverflowError:
   366:             return x
   367: 
   368:     def log_erfc(x):
   369:         return log(2) + log_ndtr(-x * 2 ** 0.5)
   370: 
   371:     def log_a_int(q, sigma, alpha):
   372:         log_a = -inf
   373:         for i in _range(alpha + 1):
   374:             log_coef_i = log(binom(alpha, i)) + i * log(q) + (alpha - i) * log(1 - q)
   375:             log_a = log_add(log_a, log_coef_i + (i * i - i) / (2 * (sigma ** 2)))
   376:         return _float(log_a)
   377: 
   378:     def log_a_frac(q, sigma, alpha):
   379:         log_a0, log_a1 = -inf, -inf
   380:         i = 0
   381:         z0 = sigma ** 2 * log(1 / q - 1) + 0.5
   382:         while True:
   383:             coef = binom(alpha, i)
   384:             log_coef = log(_abs(coef))
   385:             j = alpha - i
   386:             log_t0 = log_coef + i * log(q) + j * log(1 - q)
   387:             log_t1 = log_coef + j * log(q) + i * log(1 - q)
   388:             log_e0 = log(0.5) + log_erfc((i - z0) / (sqrt(2) * sigma))
   389:             log_e1 = log(0.5) + log_erfc((z0 - j) / (sqrt(2) * sigma))
   390:             log_s0 = log_t0 + (i * i - i) / (2 * (sigma ** 2)) + log_e0
   391:             log_s1 = log_t1 + (j * j - j) / (2 * (sigma ** 2)) + log_e1
   392:             if coef > 0:
   393:                 log_a0, log_a1 = log_add(log_a0, log_s0), log_add(log_a1, log_s1)
   394:             else:
   395:                 log_a0, log_a1 = log_sub(log_a0, log_s0), log_sub(log_a1, log_s1)
   396:             i += 1
   397:             if _max(log_s0, log_s1) < -30:
   398:                 break
   399:         return log_add(log_a0, log_a1)
   400: 
   401:     def step_rdp(q, sigma, alpha):
   402:         if q == 0:
   403:             return 0.0
   404:         if sigma == 0:
   405:             return inf
   406:         if q == 1.0:
   407:             return alpha / (2 * sigma ** 2)
   408:         if _float(alpha).is_integer():
   409:             return log_a_int(q, sigma, _int(alpha)) / (alpha - 1)
   410:         return log_a_frac(q, sigma, alpha) / (alpha - 1)
   411: 
   412:     def epsilon(schedule, q, delta, alphas=None, cache=None):
   413:         """(epsilon, best_alpha) of `schedule`, a list of (sigma, n_steps);
   414:         `cache` (a dict) keeps each sigma's one-step RDP across calls."""
   415:         q, delta = _float(q), _float(delta)
   416:         alphas = orders if alphas is None else _tuple(alphas)
   417:         cache = {} if cache is None else cache
   418:         rdp = [0.0] * _len(alphas)
   419:         for sigma, n_steps in schedule:
   420:             key = (q, _float(sigma), alphas)
   421:             per_step = cache.get(key)
   422:             if per_step is None:
   423:                 per_step = cache[key] = [step_rdp(q, key[1], a) for a in alphas]
   424:             rdp = [r + n_steps * s for r, s in _zip(rdp, per_step)]
   425:         best_eps, best_alpha = inf, None
   426:         for alpha, r in _zip(alphas, rdp):
   427:             if alpha <= 1:
   428:                 continue
   429:             eps = r - log(delta) / (alpha - 1) + log(1 - 1 / alpha)
   430:             if eps < best_eps:
   431:                 best_eps, best_alpha = eps, alpha
   432:         return _max(0, best_eps), best_alpha
   433: 
   434:     return epsilon
   435: 
   436: 
   437: _rdp_epsilon = _make_rdp_accountant()
   438: 
   439: 
   440: class _PrivacyLedger:
   441:     """Counts the steps run at each noise multiplier sigma_t the harness
   442:     applied; epsilon() composes their subsampled-Gaussian RDP per order (no
   443:     shortcut through an "effective" sigma)."""
   444: 
   445:     def __init__(self, q, delta, _float=_ORIG["builtins.float"]):
   446:         self.q = _float(q)
   447:         self.delta = _float(delta)
   448:         self.steps = 0
   449:         self.last_sigma = None
   450:         self.counts = {}  # sigma_t -> number of steps
   451:         self.rdp_cache = {}
   452: 
   453:     def record(self, sigma):
   454:         self.steps += 1
   455:         self.last_sigma = sigma
   456:         self.counts[sigma] = self.counts.get(sigma, 0) + 1
   457: 
   458:     def epsilon(self, _epsilon=_rdp_epsilon, _list=_ORIG["builtins.list"]):
   459:         return _epsilon(_list(self.counts.items()), self.q, self.delta,
   460:                         cache=self.rdp_cache)[0]
   461: 
   462: 
   463: def _positive_finite(value, name, _float=_ORIG["builtins.float"],
   464:                      _isfinite=_ORIG["math.isfinite"]):
   465:     value = _float(value)
   466:     if not _isfinite(value) or value <= 0:
   467:         raise RuntimeError(f"DPMechanism returned invalid {name}={value!r}; "
   468:                            f"it must be a finite positive number")
   469:     return value
   470: 
   471: 
   472: def privatize_step(dp_mechanism, per_sample_grads, step, epoch, ledger,
   473:                    _check=_check_harness, _positive=_positive_finite,
   474:                    _Tensor=_ORIG["torch.Tensor"], _randn_like=_ORIG["torch.randn_like"],
   475:                    _cat=_ORIG["torch.cat"], _vector_norm=_ORIG["torch.linalg.vector_norm"],
   476:                    _rtol=_CLIP_NORM_RTOL, _bool=_ORIG["builtins.bool"],
   477:                    _type=_ORIG["builtins.type"], _len=_ORIG["builtins.len"],
   478:                    _isinstance=_ORIG["builtins.isinstance"], _list=_ORIG["builtins.list"],
   479:                    _tuple=_ORIG["builtins.tuple"], _zip=_ORIG["builtins.zip"],
   480:                    _detach=_ORIG["torch.Tensor.detach"]):
   481:     """One step of the Gaussian mechanism with the mechanism's C_t and sigma_t.
   482: 
   483:     The mechanism sees a copy of the per-sample gradients and returns
   484:     per-sample multipliers plus the bound C_t; the harness scales the
   485:     original gradients, verifies ||scaled_i|| <= C_t for every sample, averages
   486:     over the batch, adds N(0, (sigma_t * C_t / B)^2) noise and records sigma_t.
   487:     """
   488:     batch_size = per_sample_grads[0].shape[0]
   489:     scale, clip_norm = dp_mechanism.clip(
   490:         [g.clone() for g in per_sample_grads], step, epoch
   491:     )
   492:     clip_norm = _positive(clip_norm, "clip_norm")
   493:     sigma = _positive(dp_mechanism.get_noise_multiplier(step, epoch),
   494:                       "noise multiplier")
   495:     scales = _list(scale) if _isinstance(scale, (_list, _tuple)) else [scale] * _len(per_sample_grads)
   496:     _check()  # no DPMechanism code runs past this point in this step: the
   497:     # returned scales are only read through `shape` and unbound Tensor methods
   498: 
   499:     if _len(scales) != _len(per_sample_grads):
   500:         raise RuntimeError(f"DPMechanism.clip returned {_len(scales)} scale tensors "

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
