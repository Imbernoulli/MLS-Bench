# MLS-Bench: optimization-variance-reduction

# Variance Reduction for Stochastic Optimization

## Research Question
Design an improved variance reduction strategy for stochastic gradient descent on finite-sum optimization problems. Your method should accelerate convergence compared to vanilla mini-batch SGD by reducing the variance of gradient estimates.

## Background
Many machine learning problems take the form of finite-sum optimization:

    min_x  F(x) = (1/n) * sum_{i=1}^{n} f_i(x)

Standard SGD uses a stochastic gradient from a random mini-batch, which has variance proportional to `1 / b` (where `b` is the batch size). Variance reduction methods use auxiliary information (snapshots, recursive corrections, momentum) to reduce this variance, enabling faster convergence — often achieving linear convergence rates for strongly convex problems where SGD only achieves sublinear rates.

Key methods in this area:
- **SVRG** — periodic full-gradient snapshot + control variate (Johnson and Zhang, "Accelerating Stochastic Gradient Descent using Predictive Variance Reduction", NeurIPS 2013).
- **SARAH** — recursive gradient correction (Nguyen, Liu, Scheinberg, and Takáč, "SARAH: A Novel Method for Machine Learning Problems Using Stochastic Recursive Gradient", ICML 2017; arXiv:1703.00102).
- **STORM** — momentum-based online variance reduction (Cutkosky and Orabona, "Momentum-Based Variance Reduction in Non-Convex SGD", NeurIPS 2019; arXiv:1905.10018).
- **STORM+** — fully adaptive STORM without smoothness/gradient-norm constants (Levy, Kavis, and Cevher, "STORM+: Fully Adaptive SGD with Recursive Momentum for Nonconvex Optimization", NeurIPS 2021; arXiv:2111.01040).
- **SPIDER / PAGE** — biased recursive estimators with optimal complexity for non-convex problems (Fang, Li, Lin, and Zhang, NeurIPS 2018; Li, Bao, Zhang, and Richtárik, ICML 2021).

## Task
Modify the `VarianceReductionOptimizer` class in `custom_vr.py` (inside the editable block). You must implement:

1. **`__init__(self, model, lr, l2_reg, loss_type, n_train, batch_size, device)`** — initialize any state needed for variance reduction (snapshot parameters, running gradient estimates, buffers, etc.).
2. **`train_one_epoch(self, X_train, y_train)`** — train for one epoch over the data, returning a dict with at least `'avg_loss'` (and optionally `'full_grad_count'` if you use full gradient computations).

The default implementation is vanilla mini-batch SGD. Your goal is to design a variance reduction mechanism that improves convergence.

## Interface

### Available helper functions (FIXED, use these for gradient computation):
```python
compute_full_gradient(model, X_train, y_train, loss_type, l2_reg, device)
# -> returns list of gradient tensors (one per parameter)

compute_stochastic_gradient(model, X_batch, y_batch, loss_type, l2_reg)
# -> returns list of gradient tensors for a mini-batch

compute_loss_on_batch(model, X_batch, y_batch, loss_type, l2_reg)
# -> returns scalar loss tensor
```

### Constraints
- You may call `compute_full_gradient` at most once per epoch.
- Parameter updates must use `p.data.add_(...)` or similar in-place operations.
- Must work across all problems with the same code.
- The learning rate (`self.lr`) and L2 regularization (`self.l2_reg`) are fixed.
- Do not modify the model architecture, loss function, or evaluation code.

## Baselines (paper-cited reference implementations)
- **svrg** — Johnson and Zhang (NeurIPS 2013); paper-default outer-loop length `m = n / b` and a single full-gradient snapshot per epoch.
- **storm** — Cutkosky and Orabona (NeurIPS 2019; arXiv:1905.10018); paper-default momentum schedule `a_t = c / (k + t)^{2/3}` with the prescribed adaptive step size.
- **storm_plus** — Levy, Kavis, and Cevher (NeurIPS 2021; arXiv:2111.01040); paper-default fully adaptive step-size and momentum without prior knowledge of smoothness or gradient-norm bounds.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/opt-vr-bench/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits that change code outside these ranges — or creating new files, or
deleting whole files — will cause your submission to be invalid.

The line numbers mark an editable **region**, not a fixed line-count budget: you
may add or remove lines inside it. Only code outside the editable ranges must
stay unchanged.

- `opt-vr-bench/custom_vr.py`
- editable lines **96–177**




## Readable Context


### `opt-vr-bench/custom_vr.py`  [EDITABLE — lines 96–177 only]

```python
     1: """Variance Reduction Benchmark for Finite-Sum Optimization (optimizer module).
     2: 
     3: This file holds the agent-editable variance-reduction optimizer together with
     4: the FIXED gradient helpers it is allowed to call. The data loaders (which build
     5: the synthetic ground-truth w_true), the test labels, the per-problem scoring
     6: (``evaluate``) and the training driver (``train_problem`` / ``main``) live in a
     7: separate FIXED module (``vr_driver.py``) that the agent's process cannot reach
     8: from here. ``vr_driver.py`` loads the PRE-GENERATED observable data and drives
     9: this optimizer, so the editable optimizer can neither name/recompute the eval
    10: datasets, read X_test/y_test, nor score against the held-out labels.
    11: 
    12: Evaluates variance reduction strategies for stochastic gradient methods on
    13: finite-sum problems:  min_x  F(x) = (1/n) * sum_{i=1}^{n} f_i(x)
    14: 
    15: Benchmarks:
    16:   1. logistic  -- L2-regularized logistic regression on MNIST (convex)
    17:   2. mlp       -- 2-layer MLP on CIFAR-10 (non-convex)
    18:   3. conditioned -- L2-regularized linear regression on synthetic
    19:                     ill-conditioned data (strongly convex)
    20: 
    21: The benchmark is run via:  python opt-vr-bench/vr_driver.py --problem <name> ...
    22: """
    23: 
    24: import math
    25: from typing import List
    26: 
    27: import numpy as np
    28: import torch
    29: import torch.nn as nn
    30: import torch.nn.functional as F
    31: 
    32: 
    33: # ============================================================================
    34: # FIXED -- Gradient helpers available to the optimizer
    35: # ============================================================================
    36: 
    37: def compute_loss_on_batch(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
    38:                           loss_type: str, l2_reg: float) -> torch.Tensor:
    39:     """Compute loss on a batch, including L2 regularization."""
    40:     pred = model(X)
    41:     if loss_type == "cross_entropy":
    42:         loss = F.cross_entropy(pred, y)
    43:     elif loss_type == "mse":
    44:         loss = F.mse_loss(pred, y)
    45:     else:
    46:         raise ValueError(f"Unknown loss type: {loss_type}")
    47:     # L2 regularization
    48:     if l2_reg > 0:
    49:         reg = sum(p.pow(2).sum() for p in model.parameters()) * l2_reg / 2
    50:         loss = loss + reg
    51:     return loss
    52: 
    53: 
    54: def compute_full_gradient(model: nn.Module, X_train: torch.Tensor,
    55:                           y_train: torch.Tensor, loss_type: str,
    56:                           l2_reg: float, device: torch.device,
    57:                           batch_size: int = 512) -> List[torch.Tensor]:
    58:     """Compute the full gradient (1/n) * sum_i grad f_i(x) over all training data.
    59: 
    60:     Returns a list of gradient tensors, one per parameter (same order as
    61:     model.parameters()).
    62:     """
    63:     model.zero_grad()
    64:     n = X_train.size(0)
    65:     # Accumulate gradient over mini-batches for memory efficiency
    66:     for start in range(0, n, batch_size):
    67:         end = min(start + batch_size, n)
    68:         Xb = X_train[start:end].to(device)
    69:         yb = y_train[start:end].to(device)
    70:         loss = compute_loss_on_batch(model, Xb, yb, loss_type, l2_reg)
    71:         # Scale by fraction of data in this batch
    72:         (loss * (end - start) / n).backward()
    73:     full_grad = [p.grad.clone() for p in model.parameters()]
    74:     model.zero_grad()
    75:     return full_grad
    76: 
    77: 
    78: def compute_stochastic_gradient(model: nn.Module, X_batch: torch.Tensor,
    79:                                 y_batch: torch.Tensor, loss_type: str,
    80:                                 l2_reg: float) -> List[torch.Tensor]:
    81:     """Compute stochastic gradient on a mini-batch.
    82: 
    83:     Returns a list of gradient tensors, one per parameter.
    84:     """
    85:     model.zero_grad()
    86:     loss = compute_loss_on_batch(model, X_batch, y_batch, loss_type, l2_reg)
    87:     loss.backward()
    88:     sg = [p.grad.clone() for p in model.parameters()]
    89:     model.zero_grad()
    90:     return sg
    91: 
    92: 
    93: # ============================================================================
    94: # EDITABLE -- Variance Reduction Strategy
    95: # ============================================================================
    96: # Design a variance reduction mechanism for stochastic gradient computation.
    97: # You may modify ONLY this section.
    98: #
    99: # Interface contract:
   100: #   - VarianceReductionOptimizer.__init__(model, lr, l2_reg, loss_type, n_train, batch_size, device)
   101: #   - VarianceReductionOptimizer.train_one_epoch(X_train, y_train)
   102: #     -> trains for one epoch, returns dict with 'avg_loss'
   103: #
   104: # Available helper functions (FIXED, defined above):
   105: #   - compute_full_gradient(model, X_train, y_train, loss_type, l2_reg, device)
   106: #     -> returns list of full gradient tensors
   107: #   - compute_stochastic_gradient(model, X_batch, y_batch, loss_type, l2_reg)
   108: #     -> returns list of stochastic gradient tensors on a mini-batch
   109: #   - compute_loss_on_batch(model, X_batch, y_batch, loss_type, l2_reg)
   110: #     -> returns scalar loss tensor
   111: #
   112: # Constraints:
   113: #   - Must work across all problems with the shared hyperparameter config
   114: #   - May use full gradient computation (compute_full_gradient) at most once
   115: #     per epoch (to maintain sublinear per-epoch cost)
   116: #   - Must respect the provided learning rate and L2 regularization
   117: #   - The model parameters should be updated in-place (via param.data)
   118: 
   119: class VarianceReductionOptimizer:
   120:     """Variance reduction strategy for finite-sum optimization.
   121: 
   122:     Default implementation: vanilla mini-batch SGD (no variance reduction).
   123:     The agent should replace this with a variance-reduced method.
   124:     """
   125: 
   126:     def __init__(self, model: nn.Module, lr: float, l2_reg: float,
   127:                  loss_type: str, n_train: int, batch_size: int,
   128:                  device: torch.device):
   129:         self.model = model
   130:         self.lr = lr
   131:         self.l2_reg = l2_reg
   132:         self.loss_type = loss_type
   133:         self.n_train = n_train
   134:         self.batch_size = batch_size
   135:         self.device = device
   136:         self.params = list(model.parameters())
   137: 
   138:     def train_one_epoch(self, X_train: torch.Tensor,
   139:                         y_train: torch.Tensor) -> dict:
   140:         """Train for one pass over the data.
   141: 
   142:         Args:
   143:             X_train: full training features [n, ...]
   144:             y_train: full training labels [n, ...]
   145: 
   146:         Returns:
   147:             dict with at least 'avg_loss' key
   148:         """
   149:         self.model.train()
   150:         n = X_train.size(0)
   151:         indices = torch.randperm(n)
   152:         total_loss = 0.0
   153:         n_batches = 0
   154: 
   155:         for start in range(0, n, self.batch_size):
   156:             end = min(start + self.batch_size, n)
   157:             idx = indices[start:end]
   158:             Xb = X_train[idx].to(self.device)
   159:             yb = y_train[idx].to(self.device)
   160: 
   161:             # Standard SGD: compute stochastic gradient and update
   162:             self.model.zero_grad()
   163:             loss = compute_loss_on_batch(
   164:                 self.model, Xb, yb, self.loss_type, self.l2_reg
   165:             )
   166:             loss.backward()
   167: 
   168:             # SGD parameter update
   169:             with torch.no_grad():
   170:                 for p in self.params:
   171:                     if p.grad is not None:
   172:                         p.data.add_(p.grad, alpha=-self.lr)
   173: 
   174:             total_loss += loss.item()
   175:             n_batches += 1
   176: 
   177:         return {"avg_loss": total_loss / max(n_batches, 1)}
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `svrg` baseline — editable region  [READ-ONLY — reference implementation]

In `opt-vr-bench/custom_vr.py`:

```python
Lines 96–218:
    93: # ============================================================================
    94: # EDITABLE -- Variance Reduction Strategy
    95: # ============================================================================
    96: # Design a variance reduction mechanism for stochastic gradient computation.
    97: # You may modify ONLY this section.
    98: #
    99: # Interface contract:
   100: #   - VarianceReductionOptimizer.__init__(model, lr, l2_reg, loss_type, n_train, batch_size, device)
   101: #   - VarianceReductionOptimizer.train_one_epoch(X_train, y_train)
   102: #     -> trains for one epoch, returns dict with 'avg_loss'
   103: #
   104: # Available helper functions (FIXED, defined above):
   105: #   - compute_full_gradient(model, X_train, y_train, loss_type, l2_reg, device)
   106: #     -> returns list of full gradient tensors
   107: #   - compute_stochastic_gradient(model, X_batch, y_batch, loss_type, l2_reg)
   108: #     -> returns list of stochastic gradient tensors on a mini-batch
   109: #   - compute_loss_on_batch(model, X_batch, y_batch, loss_type, l2_reg)
   110: #     -> returns scalar loss tensor
   111: #
   112: # Constraints:
   113: #   - Must work across all problems with the shared hyperparameter config
   114: #   - May use full gradient computation (compute_full_gradient) at most once
   115: #     per epoch (to maintain sublinear per-epoch cost)
   116: #   - Must respect the provided learning rate and L2 regularization
   117: #   - The model parameters should be updated in-place (via param.data)
   118: 
   119: class VarianceReductionOptimizer:
   120:     """SVRG with adaptive step sizing and geometric growth cap.
   121: 
   122:     At the start of each epoch, computes a full gradient at the current
   123:     snapshot point.  Each inner iteration uses the control-variate estimator:
   124:         v_t = grad_i(x_t) - grad_i(x_snap) + mu   (where mu = full_grad(x_snap))
   125: 
   126:     Step size: eta = min(lr, 0.01 * ||w||/||g||, eta_max).
   127:     eta_max grows geometrically at 1.5x per epoch, allowing the step to
   128:     increase as training progresses (gnorm decreases) while preventing the
   129:     runaway growth that caused divergence in v2.
   130:     """
   131: 
   132:     def __init__(self, model: nn.Module, lr: float, l2_reg: float,
   133:                  loss_type: str, n_train: int, batch_size: int,
   134:                  device: torch.device):
   135:         self.model = model
   136:         self.lr = lr
   137:         self.l2_reg = l2_reg
   138:         self.loss_type = loss_type
   139:         self.n_train = n_train
   140:         self.batch_size = batch_size
   141:         self.device = device
   142:         self.params = list(model.parameters())
   143:         self.snapshot_params = None
   144:         self.full_grad = None
   145:         self.eta_max = None
   146: 
   147:     def _save_snapshot(self):
   148:         self.snapshot_params = [p.data.clone() for p in self.params]
   149: 
   150:     def _load_snapshot(self):
   151:         saved = [p.data.clone() for p in self.params]
   152:         for p, sp in zip(self.params, self.snapshot_params):
   153:             p.data.copy_(sp)
   154:         return saved
   155: 
   156:     def _restore_params(self, saved):
   157:         for p, s in zip(self.params, saved):
   158:             p.data.copy_(s)
   159: 
   160:     def train_one_epoch(self, X_train: torch.Tensor,
   161:                         y_train: torch.Tensor) -> dict:
   162:         self.model.train()
   163:         n = X_train.size(0)
   164: 
   165:         # --- Snapshot ---
   166:         self._save_snapshot()
   167:         self.full_grad = compute_full_gradient(
   168:             self.model, X_train, y_train, self.loss_type,
   169:             self.l2_reg, self.device
   170:         )
   171: 
   172:         # Standard SVRG: use the provided lr directly. For ill-conditioned
   173:         # MSE problems cap the first-step magnitude by 1/||∇F|| to prevent
   174:         # divergence (previous adaptive 1.5x-geometric schedule blew up to
   175:         # eta≈1e5 and gave final MSE≈1e34).
   176:         if self.loss_type == 'mse':
   177:             gnorm = math.sqrt(sum(
   178:                 g.pow(2).sum().item() for g in self.full_grad)) + 1e-8
   179:             effective_lr = min(self.lr, 1.0 / gnorm)
   180:         else:
   181:             effective_lr = self.lr
   182: 
   183:         indices = torch.randperm(n)
   184:         total_loss = 0.0
   185:         n_batches = 0
   186: 
   187:         for start in range(0, n, self.batch_size):
   188:             end = min(start + self.batch_size, n)
   189:             idx = indices[start:end]
   190:             Xb = X_train[idx].to(self.device)
   191:             yb = y_train[idx].to(self.device)
   192: 
   193:             grad_at_x = compute_stochastic_gradient(
   194:                 self.model, Xb, yb, self.loss_type, self.l2_reg
   195:             )
   196: 
   197:             saved = self._load_snapshot()
   198:             grad_at_snap = compute_stochastic_gradient(
   199:                 self.model, Xb, yb, self.loss_type, self.l2_reg
   200:             )
   201:             self._restore_params(saved)
   202: 
   203:             # SVRG update: v = grad_i(x_t) - grad_i(x_snap) + mu
   204:             with torch.no_grad():
   205:                 for p, gx, gs, mu in zip(self.params, grad_at_x,
   206:                                          grad_at_snap, self.full_grad):
   207:                     vr_grad = gx - gs + mu
   208:                     p.data.add_(vr_grad, alpha=-effective_lr)
   209: 
   210:             with torch.no_grad():
   211:                 loss = compute_loss_on_batch(
   212:                     self.model, Xb, yb, self.loss_type, self.l2_reg
   213:                 )
   214:                 total_loss += loss.item()
   215:             n_batches += 1
   216: 
   217:         return {"avg_loss": total_loss / max(n_batches, 1),
   218:                 "full_grad_count": 1}
```

### `storm` baseline — editable region  [READ-ONLY — reference implementation]

In `opt-vr-bench/custom_vr.py`:

```python
Lines 96–222:
    93: # ============================================================================
    94: # EDITABLE -- Variance Reduction Strategy
    95: # ============================================================================
    96: # Design a variance reduction mechanism for stochastic gradient computation.
    97: # You may modify ONLY this section.
    98: #
    99: # Interface contract:
   100: #   - VarianceReductionOptimizer.__init__(model, lr, l2_reg, loss_type, n_train, batch_size, device)
   101: #   - VarianceReductionOptimizer.train_one_epoch(X_train, y_train)
   102: #     -> trains for one epoch, returns dict with 'avg_loss'
   103: #
   104: # Available helper functions (FIXED, defined above):
   105: #   - compute_full_gradient(model, X_train, y_train, loss_type, l2_reg, device)
   106: #     -> returns list of full gradient tensors
   107: #   - compute_stochastic_gradient(model, X_batch, y_batch, loss_type, l2_reg)
   108: #     -> returns list of stochastic gradient tensors on a mini-batch
   109: #   - compute_loss_on_batch(model, X_batch, y_batch, loss_type, l2_reg)
   110: #     -> returns scalar loss tensor
   111: #
   112: # Constraints:
   113: #   - Must work across all problems with the shared hyperparameter config
   114: #   - May use full gradient computation (compute_full_gradient) at most once
   115: #     per epoch (to maintain sublinear per-epoch cost)
   116: #   - Must respect the provided learning rate and L2 regularization
   117: #   - The model parameters should be updated in-place (via param.data)
   118: 
   119: class VarianceReductionOptimizer:
   120:     """STORM: STochastic Recursive Momentum.
   121: 
   122:     Maintains a momentum-based gradient estimator that achieves variance
   123:     reduction without requiring periodic full gradient computations (unlike
   124:     SVRG/SARAH).  The key idea is to use an exponential moving average of
   125:     recursively corrected stochastic gradients:
   126: 
   127:         d_t = (1-a) * g_t + a * (d_{t-1} + g_t - g_{t-1}')
   128: 
   129:     where g_t = grad_i(x_t), g_{t-1}' = grad_i(x_{t-1}), and a is a
   130:     momentum coefficient.  The first epoch uses a full gradient to warm-start.
   131:     """
   132: 
   133:     def __init__(self, model: nn.Module, lr: float, l2_reg: float,
   134:                  loss_type: str, n_train: int, batch_size: int,
   135:                  device: torch.device):
   136:         self.model = model
   137:         self.lr = lr
   138:         self.l2_reg = l2_reg
   139:         self.loss_type = loss_type
   140:         self.n_train = n_train
   141:         self.batch_size = batch_size
   142:         self.device = device
   143:         self.params = list(model.parameters())
   144:         # Momentum coefficient (STORM paper recommends a = 1 - 1/sqrt(T))
   145:         n_steps_per_epoch = max(1, n_train // batch_size)
   146:         self.momentum = 1.0 - 1.0 / math.sqrt(n_steps_per_epoch)
   147:         # Running gradient estimator
   148:         self.d = None
   149:         # Previous parameters for correction term
   150:         self.prev_params = None
   151:         self.initialized = False
   152: 
   153:     def _save_params(self):
   154:         return [p.data.clone() for p in self.params]
   155: 
   156:     def _load_params(self, saved):
   157:         for p, s in zip(self.params, saved):
   158:             p.data.copy_(s)
   159: 
   160:     def train_one_epoch(self, X_train: torch.Tensor,
   161:                         y_train: torch.Tensor) -> dict:
   162:         self.model.train()
   163:         n = X_train.size(0)
   164:         a = self.momentum
   165:         full_grad_count = 0
   166: 
   167:         # Initialize with full gradient on first epoch
   168:         if not self.initialized:
   169:             self.d = compute_full_gradient(
   170:                 self.model, X_train, y_train, self.loss_type,
   171:                 self.l2_reg, self.device
   172:             )
   173:             self.prev_params = self._save_params()
   174:             # First step using full gradient
   175:             with torch.no_grad():
   176:                 for p, di in zip(self.params, self.d):
   177:                     p.data.add_(di, alpha=-self.lr)
   178:             self.initialized = True
   179:             full_grad_count = 1
   180: 
   181:         indices = torch.randperm(n)
   182:         total_loss = 0.0
   183:         n_batches = 0
   184: 
   185:         for start in range(0, n, self.batch_size):
   186:             end = min(start + self.batch_size, n)
   187:             idx = indices[start:end]
   188:             Xb = X_train[idx].to(self.device)
   189:             yb = y_train[idx].to(self.device)
   190: 
   191:             # Current stochastic gradient g_t = grad_i(x_t)
   192:             current_params = self._save_params()
   193:             g_current = compute_stochastic_gradient(
   194:                 self.model, Xb, yb, self.loss_type, self.l2_reg
   195:             )
   196: 
   197:             # Previous stochastic gradient g_{t-1}' = grad_i(x_{t-1})
   198:             self._load_params(self.prev_params)
   199:             g_prev = compute_stochastic_gradient(
   200:                 self.model, Xb, yb, self.loss_type, self.l2_reg
   201:             )
   202:             self._load_params(current_params)
   203: 
   204:             # STORM update: d_t = (1-a)*g_t + a*(d_{t-1} + g_t - g_{t-1}')
   205:             with torch.no_grad():
   206:                 for i, (p, gc, gp, di) in enumerate(zip(
   207:                         self.params, g_current, g_prev, self.d)):
   208:                     self.d[i] = (1 - a) * gc + a * (di + gc - gp)
   209:                     p.data.add_(self.d[i], alpha=-self.lr)
   210: 
   211:             self.prev_params = self._save_params()
   212: 
   213:             # Track loss
   214:             with torch.no_grad():
   215:                 loss = compute_loss_on_batch(
   216:                     self.model, Xb, yb, self.loss_type, self.l2_reg
   217:                 )
   218:                 total_loss += loss.item()
   219:             n_batches += 1
   220: 
   221:         return {"avg_loss": total_loss / max(n_batches, 1),
   222:                 "full_grad_count": full_grad_count}
```

### `storm_plus` baseline — editable region  [READ-ONLY — reference implementation]

In `opt-vr-bench/custom_vr.py`:

```python
Lines 96–227:
    93: # ============================================================================
    94: # EDITABLE -- Variance Reduction Strategy
    95: # ============================================================================
    96: # Design a variance reduction mechanism for stochastic gradient computation.
    97: # You may modify ONLY this section.
    98: #
    99: # Interface contract:
   100: #   - VarianceReductionOptimizer.__init__(model, lr, l2_reg, loss_type, n_train, batch_size, device)
   101: #   - VarianceReductionOptimizer.train_one_epoch(X_train, y_train)
   102: #     -> trains for one epoch, returns dict with 'avg_loss'
   103: #
   104: # Available helper functions (FIXED, defined above):
   105: #   - compute_full_gradient(model, X_train, y_train, loss_type, l2_reg, device)
   106: #     -> returns list of full gradient tensors
   107: #   - compute_stochastic_gradient(model, X_batch, y_batch, loss_type, l2_reg)
   108: #     -> returns list of stochastic gradient tensors on a mini-batch
   109: #   - compute_loss_on_batch(model, X_batch, y_batch, loss_type, l2_reg)
   110: #     -> returns scalar loss tensor
   111: #
   112: # Constraints:
   113: #   - Must work across all problems with the shared hyperparameter config
   114: #   - May use full gradient computation (compute_full_gradient) at most once
   115: #     per epoch (to maintain sublinear per-epoch cost)
   116: #   - Must respect the provided learning rate and L2 regularization
   117: #   - The model parameters should be updated in-place (via param.data)
   118: 
   119: class VarianceReductionOptimizer:
   120:     """STORM+ with adaptive momentum and per-step adaptive lr.
   121: 
   122:     d_t = (1-a_t)*g_t + a_t*(d_{t-1} + g_t - g_{t-1}')
   123:     a_t = min(1 - 1/sqrt(t+1), 0.999)
   124: 
   125:     Full gradient warmstart on first epoch.
   126:     Per-step lr: min(lr, 0.01 * ||w|| / ||d||).
   127:     Gradient clipping: scale d if ||d|| > 3*||g||.
   128:     """
   129: 
   130:     def __init__(self, model: nn.Module, lr: float, l2_reg: float,
   131:                  loss_type: str, n_train: int, batch_size: int,
   132:                  device: torch.device):
   133:         self.model = model
   134:         self.lr = lr
   135:         self.l2_reg = l2_reg
   136:         self.loss_type = loss_type
   137:         self.n_train = n_train
   138:         self.batch_size = batch_size
   139:         self.device = device
   140:         self.params = list(model.parameters())
   141:         self.d = None
   142:         self.prev_params = None
   143:         self.initialized = False
   144:         self.global_step = 0
   145: 
   146:     def _save_params(self):
   147:         return [p.data.clone() for p in self.params]
   148: 
   149:     def _load_params(self, saved):
   150:         for p, s in zip(self.params, saved):
   151:             p.data.copy_(s)
   152: 
   153:     def _gnorm(self, grads):
   154:         return math.sqrt(sum(g.pow(2).sum().item() for g in grads))
   155: 
   156:     def _step_lr(self, direction):
   157:         dnorm = self._gnorm(direction)
   158:         pnorm = math.sqrt(sum(
   159:             p.data.pow(2).sum().item() for p in self.params)) + 1e-8
   160:         return min(self.lr, 0.01 * pnorm / (dnorm + 1e-8))
   161: 
   162:     def train_one_epoch(self, X_train, y_train):
   163:         self.model.train()
   164:         n = X_train.size(0)
   165:         full_grad_count = 0
   166: 
   167:         if not self.initialized:
   168:             self.d = compute_full_gradient(
   169:                 self.model, X_train, y_train, self.loss_type,
   170:                 self.l2_reg, self.device)
   171:             self.prev_params = self._save_params()
   172:             eta = self._step_lr(self.d)
   173:             with torch.no_grad():
   174:                 for p, di in zip(self.params, self.d):
   175:                     p.data.add_(di, alpha=-eta)
   176:             self.initialized = True
   177:             full_grad_count = 1
   178: 
   179:         indices = torch.randperm(n)
   180:         total_loss = 0.0
   181:         n_batches = 0
   182: 
   183:         for start in range(0, n, self.batch_size):
   184:             end = min(start + self.batch_size, n)
   185:             idx = indices[start:end]
   186:             Xb = X_train[idx].to(self.device)
   187:             yb = y_train[idx].to(self.device)
   188: 
   189:             self.global_step += 1
   190:             a = min(1.0 - 1.0 / math.sqrt(self.global_step + 1), 0.999)
   191: 
   192:             current_params = self._save_params()
   193:             g_current = compute_stochastic_gradient(
   194:                 self.model, Xb, yb, self.loss_type, self.l2_reg)
   195: 
   196:             self._load_params(self.prev_params)
   197:             g_prev = compute_stochastic_gradient(
   198:                 self.model, Xb, yb, self.loss_type, self.l2_reg)
   199:             self._load_params(current_params)
   200: 
   201:             with torch.no_grad():
   202:                 for i, (gc, gp, di) in enumerate(zip(
   203:                         g_current, g_prev, self.d)):
   204:                     self.d[i] = (1 - a) * gc + a * (di + gc - gp)
   205: 
   206:                 # Clip
   207:                 d_norm = self._gnorm(self.d)
   208:                 g_norm = self._gnorm(g_current)
   209:                 if d_norm > 3.0 * g_norm and g_norm > 1e-8:
   210:                     scale = 3.0 * g_norm / d_norm
   211:                     for di in self.d:
   212:                         di.mul_(scale)
   213: 
   214:                 eta = self._step_lr(self.d)
   215:                 for p, di in zip(self.params, self.d):
   216:                     p.data.add_(di, alpha=-eta)
   217: 
   218:             self.prev_params = self._save_params()
   219: 
   220:             with torch.no_grad():
   221:                 loss = compute_loss_on_batch(
   222:                     self.model, Xb, yb, self.loss_type, self.l2_reg)
   223:                 total_loss += loss.item()
   224:             n_batches += 1
   225: 
   226:         return {"avg_loss": total_loss / max(n_batches, 1),
   227:                 "full_grad_count": full_grad_count}
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
