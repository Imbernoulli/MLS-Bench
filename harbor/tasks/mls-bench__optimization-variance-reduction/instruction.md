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

## Evaluation
- **Problems**:
  - `logistic`: L2-regularized multinomial logistic regression on MNIST (convex, n=60K, 20 epochs).
  - `mlp`: 2-layer MLP on CIFAR-10 (non-convex, n=50K, 40 epochs).
  - `conditioned`: L2-regularized linear regression on synthetic ill-conditioned data (strongly convex, kappa=100, n=10K, 30 epochs).
- **Metrics**: `best_test_accuracy` and `final_test_accuracy` (logistic, mlp; higher is better) and `best_test_mse` / `final_test_mse` (conditioned; lower is better).
- All problems run in parallel with shared compute.

## Baselines (paper-cited reference implementations)
- **svrg** — Johnson and Zhang (NeurIPS 2013); paper-default outer-loop length `m = n / b` and a single full-gradient snapshot per epoch.
- **storm** — Cutkosky and Orabona (NeurIPS 2019; arXiv:1905.10018); paper-default momentum schedule `a_t = c / (k + t)^{2/3}` with the prescribed adaptive step size.
- **storm_plus** — Levy, Kavis, and Cevher (NeurIPS 2021; arXiv:2111.01040); paper-default fully adaptive step-size and momentum without prior knowledge of smoothness or gradient-norm bounds.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/opt-vr-bench/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `opt-vr-bench/custom_vr.py`
- editable lines **286–370**




## Readable Context


### `opt-vr-bench/custom_vr.py`  [EDITABLE — lines 286–370 only]

```python
     1: """Variance Reduction Benchmark for Finite-Sum Optimization
     2: 
     3: Evaluates variance reduction strategies for stochastic gradient methods on
     4: finite-sum problems:  min_x  F(x) = (1/n) * sum_{i=1}^{n} f_i(x)
     5: 
     6: Benchmarks:
     7:   1. logistic  -- L2-regularized logistic regression on MNIST (convex)
     8:   2. mlp       -- 2-layer MLP on CIFAR-10 (non-convex)
     9:   3. conditioned -- L2-regularized linear regression on synthetic
    10:                     ill-conditioned data (strongly convex)
    11: 
    12: Usage:
    13:   python opt-vr-bench/custom_vr.py --problem <name> \
    14:       --seed $SEED --output-dir $OUTPUT_DIR
    15: """
    16: 
    17: import argparse
    18: import math
    19: import os
    20: import time
    21: from typing import Any, Callable, Dict, List, Optional, Tuple
    22: 
    23: import numpy as np
    24: import torch
    25: import torch.nn as nn
    26: import torch.nn.functional as F
    27: from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
    28: 
    29: 
    30: # ============================================================================
    31: # FIXED -- Utilities
    32: # ============================================================================
    33: 
    34: def set_seed(seed: int):
    35:     """Set all random seeds for reproducibility."""
    36:     import random
    37:     random.seed(seed)
    38:     np.random.seed(seed)
    39:     torch.manual_seed(seed)
    40:     torch.cuda.manual_seed_all(seed)
    41:     torch.backends.cudnn.deterministic = True
    42:     torch.backends.cudnn.benchmark = False
    43: 
    44: 
    45: # ============================================================================
    46: # FIXED -- Model Definitions
    47: # ============================================================================
    48: 
    49: class LogisticRegression(nn.Module):
    50:     """Multinomial logistic regression for MNIST (convex with L2 reg)."""
    51:     def __init__(self, input_dim=784, num_classes=10):
    52:         super().__init__()
    53:         self.linear = nn.Linear(input_dim, num_classes)
    54: 
    55:     def forward(self, x):
    56:         return self.linear(x.view(x.size(0), -1))
    57: 
    58: 
    59: class SmallMLP(nn.Module):
    60:     """2-layer MLP for CIFAR-10 (non-convex)."""
    61:     def __init__(self, input_dim=3072, hidden_dim=256, num_classes=10):
    62:         super().__init__()
    63:         self.net = nn.Sequential(
    64:             nn.Linear(input_dim, hidden_dim),
    65:             nn.ReLU(),
    66:             nn.Linear(hidden_dim, hidden_dim),
    67:             nn.ReLU(),
    68:             nn.Linear(hidden_dim, num_classes),
    69:         )
    70: 
    71:     def forward(self, x):
    72:         return self.net(x.view(x.size(0), -1))
    73: 
    74: 
    75: class LinearModel(nn.Module):
    76:     """Linear model for regression (strongly convex with L2 reg)."""
    77:     def __init__(self, input_dim=50):
    78:         super().__init__()
    79:         self.linear = nn.Linear(input_dim, 1)
    80: 
    81:     def forward(self, x):
    82:         return self.linear(x)
    83: 
    84: 
    85: # ============================================================================
    86: # FIXED -- Data Loading
    87: # ============================================================================
    88: 
    89: def get_mnist_dataset(data_dir=os.environ.get("DATA_ROOT", "/data") + "/mnist"):
    90:     """Return MNIST train/test as (X, y) tensors."""
    91:     import torchvision
    92:     import torchvision.transforms as T
    93:     transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    94:     train_ds = torchvision.datasets.MNIST(data_dir, train=True, transform=transform)
    95:     test_ds = torchvision.datasets.MNIST(data_dir, train=False, transform=transform)
    96:     # Convert to tensors for finite-sum access
    97:     X_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    98:     y_train = torch.tensor([train_ds[i][1] for i in range(len(train_ds))])
    99:     X_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
   100:     y_test = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])
   101:     return X_train, y_train, X_test, y_test
   102: 
   103: 
   104: def get_cifar10_dataset(data_dir=os.environ.get("DATA_ROOT", "/data") + "/cifar"):
   105:     """Return CIFAR-10 train/test as (X, y) tensors."""
   106:     import torchvision
   107:     import torchvision.transforms as T
   108:     transform = T.Compose([T.ToTensor(), T.Normalize(
   109:         (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
   110:     train_ds = torchvision.datasets.CIFAR10(data_dir, train=True, transform=transform)
   111:     test_ds = torchvision.datasets.CIFAR10(data_dir, train=False, transform=transform)
   112:     X_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
   113:     y_train = torch.tensor([train_ds[i][1] for i in range(len(train_ds))])
   114:     X_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
   115:     y_test = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])
   116:     return X_train, y_train, X_test, y_test
   117: 
   118: 
   119: def get_conditioned_dataset(n_train=10000, n_test=2000, dim=50,
   120:                             condition_number=100, seed=0):
   121:     """Return synthetic ill-conditioned regression data."""
   122:     rng = np.random.RandomState(seed)
   123:     # Create design matrix with specified condition number
   124:     U, _, _ = np.linalg.svd(rng.randn(dim, dim), full_matrices=True)
   125:     singular_values = np.logspace(0, np.log10(condition_number), dim)
   126:     A = U @ np.diag(singular_values) @ U.T
   127:     # Generate data: y = X @ w_true + noise
   128:     w_true = rng.randn(dim, 1)
   129:     X_all = rng.randn(n_train + n_test, dim) @ A
   130:     y_all = X_all @ w_true + 0.1 * rng.randn(n_train + n_test, 1)
   131:     X_train = torch.tensor(X_all[:n_train], dtype=torch.float32)
   132:     y_train = torch.tensor(y_all[:n_train], dtype=torch.float32)
   133:     X_test = torch.tensor(X_all[n_train:], dtype=torch.float32)
   134:     y_test = torch.tensor(y_all[n_train:], dtype=torch.float32)
   135:     return X_train, y_train, X_test, y_test
   136: 
   137: 
   138: # ============================================================================
   139: # FIXED -- Problem Configurations
   140: # ============================================================================
   141: 
   142: PROBLEM_CONFIGS = {
   143:     "logistic": {
   144:         "lr": 0.1,
   145:         "l2_reg": 1e-4,
   146:         "n_epochs": 20,
   147:         "batch_size": 128,
   148:         "eval_interval": 1,
   149:         "target_metric": "test_accuracy",
   150:         "higher_is_better": True,
   151:         "loss_type": "cross_entropy",
   152:     },
   153:     "mlp": {
   154:         "lr": 0.05,
   155:         "l2_reg": 1e-4,
   156:         "n_epochs": 40,
   157:         "batch_size": 128,
   158:         "eval_interval": 2,
   159:         "target_metric": "test_accuracy",
   160:         "higher_is_better": True,
   161:         "loss_type": "cross_entropy",
   162:     },
   163:     "conditioned": {
   164:         "lr": 0.001,
   165:         "l2_reg": 1e-3,
   166:         "n_epochs": 30,
   167:         "batch_size": 128,
   168:         "eval_interval": 1,
   169:         "target_metric": "test_mse",
   170:         "higher_is_better": False,
   171:         "loss_type": "mse",
   172:     },
   173: }
   174: 
   175: 
   176: def build_model(problem: str, device: torch.device) -> nn.Module:
   177:     """Instantiate the model for a given problem."""
   178:     if problem == "logistic":
   179:         return LogisticRegression(input_dim=784, num_classes=10).to(device)
   180:     elif problem == "mlp":
   181:         return SmallMLP(input_dim=3072, hidden_dim=256, num_classes=10).to(device)
   182:     elif problem == "conditioned":
   183:         return LinearModel(input_dim=50).to(device)
   184:     else:
   185:         raise ValueError(f"Unknown problem: {problem}")
   186: 
   187: 
   188: def get_data(problem: str, seed: int):
   189:     """Return (X_train, y_train, X_test, y_test) for a problem."""
   190:     if problem == "logistic":
   191:         return get_mnist_dataset()
   192:     elif problem == "mlp":
   193:         return get_cifar10_dataset()
   194:     elif problem == "conditioned":
   195:         return get_conditioned_dataset(seed=seed)
   196:     else:
   197:         raise ValueError(f"Unknown problem: {problem}")
   198: 
   199: 
   200: def compute_loss_on_batch(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
   201:                           loss_type: str, l2_reg: float) -> torch.Tensor:
   202:     """Compute loss on a batch, including L2 regularization."""
   203:     pred = model(X)
   204:     if loss_type == "cross_entropy":
   205:         loss = F.cross_entropy(pred, y)
   206:     elif loss_type == "mse":
   207:         loss = F.mse_loss(pred, y)
   208:     else:
   209:         raise ValueError(f"Unknown loss type: {loss_type}")
   210:     # L2 regularization
   211:     if l2_reg > 0:
   212:         reg = sum(p.pow(2).sum() for p in model.parameters()) * l2_reg / 2
   213:         loss = loss + reg
   214:     return loss
   215: 
   216: 
   217: def compute_full_gradient(model: nn.Module, X_train: torch.Tensor,
   218:                           y_train: torch.Tensor, loss_type: str,
   219:                           l2_reg: float, device: torch.device,
   220:                           batch_size: int = 512) -> List[torch.Tensor]:
   221:     """Compute the full gradient (1/n) * sum_i grad f_i(x) over all training data.
   222: 
   223:     Returns a list of gradient tensors, one per parameter (same order as
   224:     model.parameters()).
   225:     """
   226:     model.zero_grad()
   227:     n = X_train.size(0)
   228:     # Accumulate gradient over mini-batches for memory efficiency
   229:     for start in range(0, n, batch_size):
   230:         end = min(start + batch_size, n)
   231:         Xb = X_train[start:end].to(device)
   232:         yb = y_train[start:end].to(device)
   233:         loss = compute_loss_on_batch(model, Xb, yb, loss_type, l2_reg)
   234:         # Scale by fraction of data in this batch
   235:         (loss * (end - start) / n).backward()
   236:     full_grad = [p.grad.clone() for p in model.parameters()]
   237:     model.zero_grad()
   238:     return full_grad
   239: 
   240: 
   241: def compute_stochastic_gradient(model: nn.Module, X_batch: torch.Tensor,
   242:                                 y_batch: torch.Tensor, loss_type: str,
   243:                                 l2_reg: float) -> List[torch.Tensor]:
   244:     """Compute stochastic gradient on a mini-batch.
   245: 
   246:     Returns a list of gradient tensors, one per parameter.
   247:     """
   248:     model.zero_grad()
   249:     loss = compute_loss_on_batch(model, X_batch, y_batch, loss_type, l2_reg)
   250:     loss.backward()
   251:     sg = [p.grad.clone() for p in model.parameters()]
   252:     model.zero_grad()
   253:     return sg
   254: 
   255: 
   256: @torch.no_grad()
   257: def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
   258:              loss_type: str, l2_reg: float, device: torch.device,
   259:              batch_size: int = 512) -> dict:
   260:     """Evaluate model on a dataset."""
   261:     model.eval()
   262:     total_loss = 0.0
   263:     correct = 0
   264:     total = 0
   265:     for start in range(0, X.size(0), batch_size):
   266:         end = min(start + batch_size, X.size(0))
   267:         Xb = X[start:end].to(device)
   268:         yb = y[start:end].to(device)
   269:         pred = model(Xb)
   270:         if loss_type == "cross_entropy":
   271:             total_loss += F.cross_entropy(pred, yb, reduction='sum').item()
   272:             correct += (pred.argmax(dim=1) == yb).sum().item()
   273:         elif loss_type == "mse":
   274:             total_loss += F.mse_loss(pred, yb, reduction='sum').item()
   275:         total += yb.size(0)
   276:     model.train()
   277:     result = {"test_loss": total_loss / total}
   278:     if loss_type == "cross_entropy":
   279:         result["test_accuracy"] = 100.0 * correct / total
   280:     elif loss_type == "mse":
   281:         result["test_mse"] = total_loss / total
   282:     return result
   283: 
   284: 
   285: # ============================================================================
   286: # EDITABLE -- Variance Reduction Strategy (lines 286-370)
   287: # ============================================================================
   288: # Design a variance reduction mechanism for stochastic gradient computation.
   289: # You may modify ONLY this section.
   290: #
   291: # Interface contract:
   292: #   - VarianceReductionOptimizer.__init__(model, lr, l2_reg, loss_type, n_train, batch_size, device)
   293: #   - VarianceReductionOptimizer.train_one_epoch(X_train, y_train)
   294: #     -> trains for one epoch, returns dict with 'avg_loss'
   295: #
   296: # Available helper functions (FIXED, defined above):
   297: #   - compute_full_gradient(model, X_train, y_train, loss_type, l2_reg, device)
   298: #     -> returns list of full gradient tensors
   299: #   - compute_stochastic_gradient(model, X_batch, y_batch, loss_type, l2_reg)
   300: #     -> returns list of stochastic gradient tensors on a mini-batch
   301: #   - compute_loss_on_batch(model, X_batch, y_batch, loss_type, l2_reg)
   302: #     -> returns scalar loss tensor
   303: #
   304: # Constraints:
   305: #   - Must work across all problems with the shared hyperparameter config
   306: #   - May use full gradient computation (compute_full_gradient) at most once
   307: #     per epoch (to maintain sublinear per-epoch cost)
   308: #   - Must respect the provided learning rate and L2 regularization
   309: #   - The model parameters should be updated in-place (via param.data)
   310: 
   311: class VarianceReductionOptimizer:
   312:     """Variance reduction strategy for finite-sum optimization.
   313: 
   314:     Default implementation: vanilla mini-batch SGD (no variance reduction).
   315:     The agent should replace this with a variance-reduced method.
   316:     """
   317: 
   318:     def __init__(self, model: nn.Module, lr: float, l2_reg: float,
   319:                  loss_type: str, n_train: int, batch_size: int,
   320:                  device: torch.device):
   321:         self.model = model
   322:         self.lr = lr
   323:         self.l2_reg = l2_reg
   324:         self.loss_type = loss_type
   325:         self.n_train = n_train
   326:         self.batch_size = batch_size
   327:         self.device = device
   328:         self.params = list(model.parameters())
   329: 
   330:     def train_one_epoch(self, X_train: torch.Tensor,
   331:                         y_train: torch.Tensor) -> dict:
   332:         """Train for one pass over the data.
   333: 
   334:         Args:
   335:             X_train: full training features [n, ...]
   336:             y_train: full training labels [n, ...]
   337: 
   338:         Returns:
   339:             dict with at least 'avg_loss' key
   340:         """
   341:         self.model.train()
   342:         n = X_train.size(0)
   343:         indices = torch.randperm(n)
   344:         total_loss = 0.0
   345:         n_batches = 0
   346: 
   347:         for start in range(0, n, self.batch_size):
   348:             end = min(start + self.batch_size, n)
   349:             idx = indices[start:end]
   350:             Xb = X_train[idx].to(self.device)
   351:             yb = y_train[idx].to(self.device)
   352: 
   353:             # Standard SGD: compute stochastic gradient and update
   354:             self.model.zero_grad()
   355:             loss = compute_loss_on_batch(
   356:                 self.model, Xb, yb, self.loss_type, self.l2_reg
   357:             )
   358:             loss.backward()
   359: 
   360:             # SGD parameter update
   361:             with torch.no_grad():
   362:                 for p in self.params:
   363:                     if p.grad is not None:
   364:                         p.data.add_(p.grad, alpha=-self.lr)
   365: 
   366:             total_loss += loss.item()
   367:             n_batches += 1
   368: 
   369:         return {"avg_loss": total_loss / max(n_batches, 1)}
   370: 
   371: 
   372: # ============================================================================
   373: # FIXED -- Training Driver
   374: # ============================================================================
   375: 
   376: def train_problem(problem: str, seed: int, output_dir: str):
   377:     """Train on a single problem and report metrics."""
   378:     cfg = PROBLEM_CONFIGS[problem]
   379:     set_seed(seed)
   380: 
   381:     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   382:     print(f"=== Problem: {problem} | Seed: {seed} | Device: {device} ===",
   383:           flush=True)
   384: 
   385:     # Build model
   386:     model = build_model(problem, device)
   387:     num_params = sum(p.numel() for p in model.parameters())
   388:     print(f"Model parameters: {num_params:,}", flush=True)
   389: 
   390:     # Load data
   391:     X_train, y_train, X_test, y_test = get_data(problem, seed)
   392:     n_train = X_train.size(0)
   393:     print(f"Training samples: {n_train}", flush=True)
   394: 
   395:     # Create variance reduction optimizer
   396:     optimizer = VarianceReductionOptimizer(
   397:         model=model,
   398:         lr=cfg["lr"],
   399:         l2_reg=cfg["l2_reg"],
   400:         loss_type=cfg["loss_type"],
   401:         n_train=n_train,
   402:         batch_size=cfg["batch_size"],
   403:         device=device,
   404:     )
   405: 
   406:     # Training loop (epoch-based)
   407:     best_metric = None
   408:     total_grad_comps = 0  # Track gradient computation cost
   409: 
   410:     for epoch in range(1, cfg["n_epochs"] + 1):
   411:         t0 = time.time()
   412:         train_info = optimizer.train_one_epoch(X_train, y_train)
   413:         epoch_time = time.time() - t0
   414: 
   415:         # Count approximate gradient computations
   416:         # One epoch of SGD: n/batch_size mini-batch gradient computations
   417:         # Full gradient: equivalent to n/batch_size computations
   418:         n_sgd_steps = math.ceil(n_train / cfg["batch_size"])
   419:         total_grad_comps += n_sgd_steps
   420: 
   421:         avg_loss = train_info.get("avg_loss", 0.0)
   422:         extra_full_grads = train_info.get("full_grad_count", 0)
   423:         total_grad_comps += extra_full_grads * n_sgd_steps
   424: 
   425:         print(f"TRAIN_METRICS: epoch={epoch} avg_loss={avg_loss:.6f} "
   426:               f"time={epoch_time:.2f}s grad_comps={total_grad_comps}",
   427:               flush=True)
   428: 
   429:         # Evaluation
   430:         if epoch % cfg["eval_interval"] == 0 or epoch == cfg["n_epochs"]:
   431:             metrics = evaluate(model, X_test, y_test, cfg["loss_type"],
   432:                                cfg["l2_reg"], device)
   433:             metric_val = metrics[cfg["target_metric"]]
   434: 
   435:             if best_metric is None:
   436:                 best_metric = metric_val
   437:             elif cfg["higher_is_better"]:
   438:                 best_metric = max(best_metric, metric_val)
   439:             else:
   440:                 best_metric = min(best_metric, metric_val)
   441: 
   442:             metric_str = " ".join(f"{k}={v:.6f}" for k, v in metrics.items())
   443:             print(f"EVAL_METRICS: epoch={epoch} {metric_str} "
   444:                   f"best_{cfg['target_metric']}={best_metric:.6f}",
   445:                   flush=True)
   446: 
   447:     # Final reporting
   448:     final_metrics = evaluate(model, X_test, y_test, cfg["loss_type"],
   449:                              cfg["l2_reg"], device)
   450:     final_val = final_metrics[cfg["target_metric"]]
   451: 
   452:     print(f"TEST_METRICS: "
   453:           f"best_{cfg['target_metric']}={best_metric:.6f} "
   454:           f"final_{cfg['target_metric']}={final_val:.6f} "
   455:           f"total_grad_comps={total_grad_comps}",
   456:           flush=True)
   457: 
   458:     # Save results
   459:     os.makedirs(output_dir, exist_ok=True)
   460:     result = {
   461:         "problem": problem,
   462:         "seed": seed,
   463:         "best_metric": best_metric,
   464:         "final_metric": final_val,
   465:         "target_metric": cfg["target_metric"],
   466:         "total_grad_comps": total_grad_comps,
   467:     }
   468:     torch.save(result, os.path.join(output_dir, f"result_{problem}.pt"))
   469: 
   470: 
   471: # ============================================================================
   472: # FIXED -- Main Entry Point
   473: # ============================================================================
   474: 
   475: def main():
   476:     parser = argparse.ArgumentParser(
   477:         description="Variance Reduction Benchmark for Finite-Sum Optimization")
   478:     parser.add_argument("--problem", type=str, required=True,
   479:                         choices=["logistic", "mlp", "conditioned"],
   480:                         help="Problem to run")
   481:     parser.add_argument("--seed", type=int, default=42, help="Random seed")
   482:     parser.add_argument("--output-dir", type=str, default="./results",
   483:                         help="Directory to save results")
   484:     args = parser.parse_args()
   485:     train_problem(args.problem, args.seed, args.output_dir)
   486: 
   487: 
   488: if __name__ == "__main__":
   489:     main()
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
Lines 286–408:
   283: 
   284: 
   285: # ============================================================================
   286: # Design a variance reduction mechanism for stochastic gradient computation.
   287: # You may modify ONLY this section.
   288: #
   289: # Interface contract:
   290: #   - VarianceReductionOptimizer.__init__(model, lr, l2_reg, loss_type, n_train, batch_size, device)
   291: #   - VarianceReductionOptimizer.train_one_epoch(X_train, y_train)
   292: #     -> trains for one epoch, returns dict with 'avg_loss'
   293: #
   294: # Available helper functions (FIXED, defined above):
   295: #   - compute_full_gradient(model, X_train, y_train, loss_type, l2_reg, device)
   296: #     -> returns list of full gradient tensors
   297: #   - compute_stochastic_gradient(model, X_batch, y_batch, loss_type, l2_reg)
   298: #     -> returns list of stochastic gradient tensors on a mini-batch
   299: #   - compute_loss_on_batch(model, X_batch, y_batch, loss_type, l2_reg)
   300: #     -> returns scalar loss tensor
   301: #
   302: # Constraints:
   303: #   - Must work across all problems with the shared hyperparameter config
   304: #   - May use full gradient computation (compute_full_gradient) at most once
   305: #     per epoch (to maintain sublinear per-epoch cost)
   306: #   - Must respect the provided learning rate and L2 regularization
   307: #   - The model parameters should be updated in-place (via param.data)
   308: 
   309: class VarianceReductionOptimizer:
   310:     """SVRG with adaptive step sizing and geometric growth cap.
   311: 
   312:     At the start of each epoch, computes a full gradient at the current
   313:     snapshot point.  Each inner iteration uses the control-variate estimator:
   314:         v_t = grad_i(x_t) - grad_i(x_snap) + mu   (where mu = full_grad(x_snap))
   315: 
   316:     Step size: eta = min(lr, 0.01 * ||w||/||g||, eta_max).
   317:     eta_max grows geometrically at 1.5x per epoch, allowing the step to
   318:     increase as training progresses (gnorm decreases) while preventing the
   319:     runaway growth that caused divergence in v2.
   320:     """
   321: 
   322:     def __init__(self, model: nn.Module, lr: float, l2_reg: float,
   323:                  loss_type: str, n_train: int, batch_size: int,
   324:                  device: torch.device):
   325:         self.model = model
   326:         self.lr = lr
   327:         self.l2_reg = l2_reg
   328:         self.loss_type = loss_type
   329:         self.n_train = n_train
   330:         self.batch_size = batch_size
   331:         self.device = device
   332:         self.params = list(model.parameters())
   333:         self.snapshot_params = None
   334:         self.full_grad = None
   335:         self.eta_max = None
   336: 
   337:     def _save_snapshot(self):
   338:         self.snapshot_params = [p.data.clone() for p in self.params]
   339: 
   340:     def _load_snapshot(self):
   341:         saved = [p.data.clone() for p in self.params]
   342:         for p, sp in zip(self.params, self.snapshot_params):
   343:             p.data.copy_(sp)
   344:         return saved
   345: 
   346:     def _restore_params(self, saved):
   347:         for p, s in zip(self.params, saved):
   348:             p.data.copy_(s)
   349: 
   350:     def train_one_epoch(self, X_train: torch.Tensor,
   351:                         y_train: torch.Tensor) -> dict:
   352:         self.model.train()
   353:         n = X_train.size(0)
   354: 
   355:         # --- Snapshot ---
   356:         self._save_snapshot()
   357:         self.full_grad = compute_full_gradient(
   358:             self.model, X_train, y_train, self.loss_type,
   359:             self.l2_reg, self.device
   360:         )
   361: 
   362:         # Standard SVRG: use the provided lr directly. For ill-conditioned
   363:         # MSE problems cap the first-step magnitude by 1/||∇F|| to prevent
   364:         # divergence (previous adaptive 1.5x-geometric schedule blew up to
   365:         # eta≈1e5 and gave final MSE≈1e34).
   366:         if self.loss_type == 'mse':
   367:             gnorm = math.sqrt(sum(
   368:                 g.pow(2).sum().item() for g in self.full_grad)) + 1e-8
   369:             effective_lr = min(self.lr, 1.0 / gnorm)
   370:         else:
   371:             effective_lr = self.lr
   372: 
   373:         indices = torch.randperm(n)
   374:         total_loss = 0.0
   375:         n_batches = 0
   376: 
   377:         for start in range(0, n, self.batch_size):
   378:             end = min(start + self.batch_size, n)
   379:             idx = indices[start:end]
   380:             Xb = X_train[idx].to(self.device)
   381:             yb = y_train[idx].to(self.device)
   382: 
   383:             grad_at_x = compute_stochastic_gradient(
   384:                 self.model, Xb, yb, self.loss_type, self.l2_reg
   385:             )
   386: 
   387:             saved = self._load_snapshot()
   388:             grad_at_snap = compute_stochastic_gradient(
   389:                 self.model, Xb, yb, self.loss_type, self.l2_reg
   390:             )
   391:             self._restore_params(saved)
   392: 
   393:             # SVRG update: v = grad_i(x_t) - grad_i(x_snap) + mu
   394:             with torch.no_grad():
   395:                 for p, gx, gs, mu in zip(self.params, grad_at_x,
   396:                                          grad_at_snap, self.full_grad):
   397:                     vr_grad = gx - gs + mu
   398:                     p.data.add_(vr_grad, alpha=-effective_lr)
   399: 
   400:             with torch.no_grad():
   401:                 loss = compute_loss_on_batch(
   402:                     self.model, Xb, yb, self.loss_type, self.l2_reg
   403:                 )
   404:                 total_loss += loss.item()
   405:             n_batches += 1
   406: 
   407:         return {"avg_loss": total_loss / max(n_batches, 1),
   408:                 "full_grad_count": 1}
   409: 
   410: # ============================================================================
   411: # FIXED -- Training Driver
```

### `storm` baseline — editable region  [READ-ONLY — reference implementation]

In `opt-vr-bench/custom_vr.py`:

```python
Lines 286–412:
   283: 
   284: 
   285: # ============================================================================
   286: # Design a variance reduction mechanism for stochastic gradient computation.
   287: # You may modify ONLY this section.
   288: #
   289: # Interface contract:
   290: #   - VarianceReductionOptimizer.__init__(model, lr, l2_reg, loss_type, n_train, batch_size, device)
   291: #   - VarianceReductionOptimizer.train_one_epoch(X_train, y_train)
   292: #     -> trains for one epoch, returns dict with 'avg_loss'
   293: #
   294: # Available helper functions (FIXED, defined above):
   295: #   - compute_full_gradient(model, X_train, y_train, loss_type, l2_reg, device)
   296: #     -> returns list of full gradient tensors
   297: #   - compute_stochastic_gradient(model, X_batch, y_batch, loss_type, l2_reg)
   298: #     -> returns list of stochastic gradient tensors on a mini-batch
   299: #   - compute_loss_on_batch(model, X_batch, y_batch, loss_type, l2_reg)
   300: #     -> returns scalar loss tensor
   301: #
   302: # Constraints:
   303: #   - Must work across all problems with the shared hyperparameter config
   304: #   - May use full gradient computation (compute_full_gradient) at most once
   305: #     per epoch (to maintain sublinear per-epoch cost)
   306: #   - Must respect the provided learning rate and L2 regularization
   307: #   - The model parameters should be updated in-place (via param.data)
   308: 
   309: class VarianceReductionOptimizer:
   310:     """STORM: STochastic Recursive Momentum.
   311: 
   312:     Maintains a momentum-based gradient estimator that achieves variance
   313:     reduction without requiring periodic full gradient computations (unlike
   314:     SVRG/SARAH).  The key idea is to use an exponential moving average of
   315:     recursively corrected stochastic gradients:
   316: 
   317:         d_t = (1-a) * g_t + a * (d_{t-1} + g_t - g_{t-1}')
   318: 
   319:     where g_t = grad_i(x_t), g_{t-1}' = grad_i(x_{t-1}), and a is a
   320:     momentum coefficient.  The first epoch uses a full gradient to warm-start.
   321:     """
   322: 
   323:     def __init__(self, model: nn.Module, lr: float, l2_reg: float,
   324:                  loss_type: str, n_train: int, batch_size: int,
   325:                  device: torch.device):
   326:         self.model = model
   327:         self.lr = lr
   328:         self.l2_reg = l2_reg
   329:         self.loss_type = loss_type
   330:         self.n_train = n_train
   331:         self.batch_size = batch_size
   332:         self.device = device
   333:         self.params = list(model.parameters())
   334:         # Momentum coefficient (STORM paper recommends a = 1 - 1/sqrt(T))
   335:         n_steps_per_epoch = max(1, n_train // batch_size)
   336:         self.momentum = 1.0 - 1.0 / math.sqrt(n_steps_per_epoch)
   337:         # Running gradient estimator
   338:         self.d = None
   339:         # Previous parameters for correction term
   340:         self.prev_params = None
   341:         self.initialized = False
   342: 
   343:     def _save_params(self):
   344:         return [p.data.clone() for p in self.params]
   345: 
   346:     def _load_params(self, saved):
   347:         for p, s in zip(self.params, saved):
   348:             p.data.copy_(s)
   349: 
   350:     def train_one_epoch(self, X_train: torch.Tensor,
   351:                         y_train: torch.Tensor) -> dict:
   352:         self.model.train()
   353:         n = X_train.size(0)
   354:         a = self.momentum
   355:         full_grad_count = 0
   356: 
   357:         # Initialize with full gradient on first epoch
   358:         if not self.initialized:
   359:             self.d = compute_full_gradient(
   360:                 self.model, X_train, y_train, self.loss_type,
   361:                 self.l2_reg, self.device
   362:             )
   363:             self.prev_params = self._save_params()
   364:             # First step using full gradient
   365:             with torch.no_grad():
   366:                 for p, di in zip(self.params, self.d):
   367:                     p.data.add_(di, alpha=-self.lr)
   368:             self.initialized = True
   369:             full_grad_count = 1
   370: 
   371:         indices = torch.randperm(n)
   372:         total_loss = 0.0
   373:         n_batches = 0
   374: 
   375:         for start in range(0, n, self.batch_size):
   376:             end = min(start + self.batch_size, n)
   377:             idx = indices[start:end]
   378:             Xb = X_train[idx].to(self.device)
   379:             yb = y_train[idx].to(self.device)
   380: 
   381:             # Current stochastic gradient g_t = grad_i(x_t)
   382:             current_params = self._save_params()
   383:             g_current = compute_stochastic_gradient(
   384:                 self.model, Xb, yb, self.loss_type, self.l2_reg
   385:             )
   386: 
   387:             # Previous stochastic gradient g_{t-1}' = grad_i(x_{t-1})
   388:             self._load_params(self.prev_params)
   389:             g_prev = compute_stochastic_gradient(
   390:                 self.model, Xb, yb, self.loss_type, self.l2_reg
   391:             )
   392:             self._load_params(current_params)
   393: 
   394:             # STORM update: d_t = (1-a)*g_t + a*(d_{t-1} + g_t - g_{t-1}')
   395:             with torch.no_grad():
   396:                 for i, (p, gc, gp, di) in enumerate(zip(
   397:                         self.params, g_current, g_prev, self.d)):
   398:                     self.d[i] = (1 - a) * gc + a * (di + gc - gp)
   399:                     p.data.add_(self.d[i], alpha=-self.lr)
   400: 
   401:             self.prev_params = self._save_params()
   402: 
   403:             # Track loss
   404:             with torch.no_grad():
   405:                 loss = compute_loss_on_batch(
   406:                     self.model, Xb, yb, self.loss_type, self.l2_reg
   407:                 )
   408:                 total_loss += loss.item()
   409:             n_batches += 1
   410: 
   411:         return {"avg_loss": total_loss / max(n_batches, 1),
   412:                 "full_grad_count": full_grad_count}
   413: 
   414: # ============================================================================
   415: # FIXED -- Training Driver
```

### `storm_plus` baseline — editable region  [READ-ONLY — reference implementation]

In `opt-vr-bench/custom_vr.py`:

```python
Lines 286–417:
   283: 
   284: 
   285: # ============================================================================
   286: # Design a variance reduction mechanism for stochastic gradient computation.
   287: # You may modify ONLY this section.
   288: #
   289: # Interface contract:
   290: #   - VarianceReductionOptimizer.__init__(model, lr, l2_reg, loss_type, n_train, batch_size, device)
   291: #   - VarianceReductionOptimizer.train_one_epoch(X_train, y_train)
   292: #     -> trains for one epoch, returns dict with 'avg_loss'
   293: #
   294: # Available helper functions (FIXED, defined above):
   295: #   - compute_full_gradient(model, X_train, y_train, loss_type, l2_reg, device)
   296: #     -> returns list of full gradient tensors
   297: #   - compute_stochastic_gradient(model, X_batch, y_batch, loss_type, l2_reg)
   298: #     -> returns list of stochastic gradient tensors on a mini-batch
   299: #   - compute_loss_on_batch(model, X_batch, y_batch, loss_type, l2_reg)
   300: #     -> returns scalar loss tensor
   301: #
   302: # Constraints:
   303: #   - Must work across all problems with the shared hyperparameter config
   304: #   - May use full gradient computation (compute_full_gradient) at most once
   305: #     per epoch (to maintain sublinear per-epoch cost)
   306: #   - Must respect the provided learning rate and L2 regularization
   307: #   - The model parameters should be updated in-place (via param.data)
   308: 
   309: class VarianceReductionOptimizer:
   310:     """STORM+ with adaptive momentum and per-step adaptive lr.
   311: 
   312:     d_t = (1-a_t)*g_t + a_t*(d_{t-1} + g_t - g_{t-1}')
   313:     a_t = min(1 - 1/sqrt(t+1), 0.999)
   314: 
   315:     Full gradient warmstart on first epoch.
   316:     Per-step lr: min(lr, 0.01 * ||w|| / ||d||).
   317:     Gradient clipping: scale d if ||d|| > 3*||g||.
   318:     """
   319: 
   320:     def __init__(self, model: nn.Module, lr: float, l2_reg: float,
   321:                  loss_type: str, n_train: int, batch_size: int,
   322:                  device: torch.device):
   323:         self.model = model
   324:         self.lr = lr
   325:         self.l2_reg = l2_reg
   326:         self.loss_type = loss_type
   327:         self.n_train = n_train
   328:         self.batch_size = batch_size
   329:         self.device = device
   330:         self.params = list(model.parameters())
   331:         self.d = None
   332:         self.prev_params = None
   333:         self.initialized = False
   334:         self.global_step = 0
   335: 
   336:     def _save_params(self):
   337:         return [p.data.clone() for p in self.params]
   338: 
   339:     def _load_params(self, saved):
   340:         for p, s in zip(self.params, saved):
   341:             p.data.copy_(s)
   342: 
   343:     def _gnorm(self, grads):
   344:         return math.sqrt(sum(g.pow(2).sum().item() for g in grads))
   345: 
   346:     def _step_lr(self, direction):
   347:         dnorm = self._gnorm(direction)
   348:         pnorm = math.sqrt(sum(
   349:             p.data.pow(2).sum().item() for p in self.params)) + 1e-8
   350:         return min(self.lr, 0.01 * pnorm / (dnorm + 1e-8))
   351: 
   352:     def train_one_epoch(self, X_train, y_train):
   353:         self.model.train()
   354:         n = X_train.size(0)
   355:         full_grad_count = 0
   356: 
   357:         if not self.initialized:
   358:             self.d = compute_full_gradient(
   359:                 self.model, X_train, y_train, self.loss_type,
   360:                 self.l2_reg, self.device)
   361:             self.prev_params = self._save_params()
   362:             eta = self._step_lr(self.d)
   363:             with torch.no_grad():
   364:                 for p, di in zip(self.params, self.d):
   365:                     p.data.add_(di, alpha=-eta)
   366:             self.initialized = True
   367:             full_grad_count = 1
   368: 
   369:         indices = torch.randperm(n)
   370:         total_loss = 0.0
   371:         n_batches = 0
   372: 
   373:         for start in range(0, n, self.batch_size):
   374:             end = min(start + self.batch_size, n)
   375:             idx = indices[start:end]
   376:             Xb = X_train[idx].to(self.device)
   377:             yb = y_train[idx].to(self.device)
   378: 
   379:             self.global_step += 1
   380:             a = min(1.0 - 1.0 / math.sqrt(self.global_step + 1), 0.999)
   381: 
   382:             current_params = self._save_params()
   383:             g_current = compute_stochastic_gradient(
   384:                 self.model, Xb, yb, self.loss_type, self.l2_reg)
   385: 
   386:             self._load_params(self.prev_params)
   387:             g_prev = compute_stochastic_gradient(
   388:                 self.model, Xb, yb, self.loss_type, self.l2_reg)
   389:             self._load_params(current_params)
   390: 
   391:             with torch.no_grad():
   392:                 for i, (gc, gp, di) in enumerate(zip(
   393:                         g_current, g_prev, self.d)):
   394:                     self.d[i] = (1 - a) * gc + a * (di + gc - gp)
   395: 
   396:                 # Clip
   397:                 d_norm = self._gnorm(self.d)
   398:                 g_norm = self._gnorm(g_current)
   399:                 if d_norm > 3.0 * g_norm and g_norm > 1e-8:
   400:                     scale = 3.0 * g_norm / d_norm
   401:                     for di in self.d:
   402:                         di.mul_(scale)
   403: 
   404:                 eta = self._step_lr(self.d)
   405:                 for p, di in zip(self.params, self.d):
   406:                     p.data.add_(di, alpha=-eta)
   407: 
   408:             self.prev_params = self._save_params()
   409: 
   410:             with torch.no_grad():
   411:                 loss = compute_loss_on_batch(
   412:                     self.model, Xb, yb, self.loss_type, self.l2_reg)
   413:                 total_loss += loss.item()
   414:             n_batches += 1
   415: 
   416:         return {"avg_loss": total_loss / max(n_batches, 1),
   417:                 "full_grad_count": full_grad_count}
   418: 
   419: # ============================================================================
   420: # FIXED -- Training Driver
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
