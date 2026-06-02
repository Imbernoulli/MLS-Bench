# MLS-Bench: optimization-bilevel

# Optimization Bilevel

## Research Question
Can you design a single first-order update rule that makes a fixed bilevel-optimization benchmark — Shen and Chen's penalty-based bilevel gradient descent setting — converge faster on a numerical toy and recover more of the clean data in a hyper-cleaning task?

## Background
A bilevel problem couples an outer objective `f(x, y)` to an inner problem `min_y g(x, y)` whose solution depends on `x`. Penalty-based bilevel gradient descent (PBGD) replaces the inner argmin with a penalty term that constrains the lower-level value gap and then performs first-order updates jointly on `x` and `y`. Two PBGD variants are studied in the reference work:

- **V-PBGD** uses a value-function penalty `g(x, y) - g*(x)` and is the main method of Shen and Chen, "On Penalty-based Bilevel Gradient Descent Method" (ICML 2023; arXiv:2302.05185).
- **G-PBGD** penalizes the squared lower-level gradient norm; an iterative-differentiation baseline (RHG / T-RHG) competes via inner unrolling.

The reference repository `hanshen95/penalized-bilevel-gradient-descent` provides the Section 5/6 toy convergence experiment and the data hyper-cleaning experiment, where a fraction of training labels are corrupted and the outer problem learns per-example weights so that an inner classifier trained on the weighted set generalizes to clean validation data.

## What You Can Modify
Edit only `penalized-bilevel-gradient-descent/mlsbench/custom_strategy.py` inside the editable block. Define:

1. `algorithm(state, hparams, grad_fns)` — one shared update used by both toy and data hyper-cleaning runs. It receives the current state dict, a task hparam dict, and gradient-function callables, and returns the updated state after one outer (or method-equivalent) update.
2. `TOY_HPARAMS` — scalar knobs for toy convergence.
3. `HYPERCLEAN_HPARAMS` — scalar knobs for hyper-cleaning; may contain separate `linear` and `mlp` sub-dicts.

For toy mode, `grad_fns` provides `f`, `df`, `g`, `dg_dy`, `dg_dl`, `proj`, and `init_state`. `df` is the outer gradient, `dg_dy` and `dg_dl` are inner gradients with respect to the lower variable `y` and upper variable, and `proj` projects the upper variable onto the feasible set.

For hyper-cleaning mode, `grad_fns` provides `outer_grad`, `inner_grad`, `inner_val`, and `init_state`, exposing first-order information for the validation loss, weighted training loss, and initial state.

The fixed scaffold also exposes reference helpers `run_v_pbgd(...)`, `run_g_pbgd(...)`, and `run_rhg_family(...)`. You may call them, wrap them, or implement your own update logic on top of the provided state and gradients.

The driver, dataset split, pollution protocol, metrics, and model architectures are fixed.

## Fixed Setup
### Toy / Numerical Verification
- Problem definition follows Section 5.1 / 6.1 of Shen and Chen (2023).
- Upper variable `x` is projected to `[0, 3]`.
- Multiple random initial points are sampled (per the official toy script).

### Data Hyper-Cleaning
- A standard image classification dataset is used with a train/validation/test split.
- A fixed fraction of training labels are corrupted (per the official code).
- Models: linear classifier and 2-layer MLP with sigmoid hidden layer.

## Reference Files (read-only)
- `penalized-bilevel-gradient-descent/V-PBGD/toy/toy.py`
- `penalized-bilevel-gradient-descent/V-PBGD/data-hyper-cleaning/data_hyper_clean.py`
- `penalized-bilevel-gradient-descent/G-PBGD/data_hyper_clean_gpbgd.py`
- `penalized-bilevel-gradient-descent/RHG/data_hyper_clean_rhg.py`
- `penalized-bilevel-gradient-descent/RHG/hypergrad/hypergradients.py`

## Baselines (cited reference implementations, all from `hanshen95/penalized-bilevel-gradient-descent`)
- **V-PBGD** — value-function PBGD (Shen and Chen, ICML 2023; arXiv:2302.05185).
- **G-PBGD** — gradient-norm PBGD variant from the same repo.
- **RHG** — Reverse-mode Hyper-Gradient via inner unrolling (Franceschi et al.; used as a baseline by Shen and Chen).
- **T-RHG** — Truncated RHG, the truncated-unroll baseline implemented in the same repo.

## Evaluation
Each command prints structured `TRAIN_METRICS` and `FINAL_METRICS` lines; the parser records final metrics per command label.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/penalized-bilevel-gradient-descent/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `penalized-bilevel-gradient-descent/mlsbench/custom_strategy.py`
- editable lines **227–262**


Other files you may **read** for context (do not modify):
- `penalized-bilevel-gradient-descent/RHG/hypergrad/hypergradients.py`


## Readable Context


### `penalized-bilevel-gradient-descent/mlsbench/custom_strategy.py`  [EDITABLE — lines 227–262 only]

```python
     1: """Optimization-bilevel scaffold for MLS-Bench.
     2: 
     3: The fixed driver reproduces the numerical verification and data hyper-cleaning
     4: experiments from Shen and Chen, "On Penalty-based Bilevel Gradient Descent
     5: Method" (ICML 2023 / Mathematical Programming 2025) while exposing only the
     6: method choice and official hyperparameters as editable strategy hooks.
     7: """
     8: 
     9: from __future__ import annotations
    10: 
    11: import argparse
    12: import json
    13: import math
    14: import os
    15: import random
    16: import sys
    17: import time
    18: from dataclasses import asdict, dataclass
    19: from pathlib import Path
    20: 
    21: _DATA_ROOT = os.environ.get("DATA_ROOT", "/data")
    22: 
    23: import torch
    24: import torch.nn as nn
    25: import torch.nn.functional as F
    26: from torchvision import datasets
    27: 
    28: ROOT = Path(__file__).resolve().parents[1]
    29: RHG_ROOT = ROOT / "RHG"
    30: if str(RHG_ROOT) not in sys.path:
    31:     sys.path.insert(0, str(RHG_ROOT))
    32: 
    33: try:
    34:     import hypergrad as hg
    35: except ModuleNotFoundError:
    36:     hg = None
    37: 
    38: if __name__ not in sys.modules:
    39:     import types
    40: 
    41:     _module_for_dataclasses = types.ModuleType(__name__)
    42:     _module_for_dataclasses.__dict__.update(globals())
    43:     sys.modules[__name__] = _module_for_dataclasses
    44: 
    45: 
    46: # =====================================================================
    47: # FIXED: Benchmark configuration
    48: # =====================================================================
    49: @dataclass(frozen=True)
    50: class ToyProblemConfig:
    51:     x_lower: float = 0.0
    52:     x_upper: float = 3.0
    53:     init_x_lower: float = 0.0
    54:     init_x_upper: float = 3.5
    55:     init_y_lower: float = -5.0
    56:     init_y_upper: float = 8.5
    57:     num_runs: int = 1000
    58:     stationarity_tol: float = 1e-5
    59:     residual_tol: float = 5e-2
    60:     max_steps_per_gamma: int = 20_000
    61: 
    62: 
    63: @dataclass(frozen=True)
    64: class HypercleanConfig:
    65:     dataset_name: str = "MNIST"
    66:     dataset_root: str = _DATA_ROOT + "/mnist"
    67:     train_size: int = 5000
    68:     val_size: int = 5000
    69:     test_size: int = 10000
    70:     pollute_rate: float = 0.5
    71: 
    72: 
    73: @dataclass(frozen=True)
    74: class ToyStrategy:
    75:     method: str
    76:     gams: tuple[float, ...]
    77:     alpha0: float
    78: 
    79:     def validate(self) -> None:
    80:         if self.method not in {"v_pbgd", "g_pbgd"}:
    81:             raise ValueError(f"Unsupported toy method: {self.method}")
    82:         if not self.gams:
    83:             raise ValueError("Toy strategy must contain at least one penalty value.")
    84:         if any(gam <= 0.0 for gam in self.gams):
    85:             raise ValueError(f"Penalty values must be positive, got {self.gams}")
    86:         if self.alpha0 <= 0.0:
    87:             raise ValueError("alpha0 must be positive.")
    88: 
    89: 
    90: @dataclass(frozen=True)
    91: class HypercleanStrategy:
    92:     method: str
    93:     lrx: float = 0.0
    94:     lry: float = 0.0
    95:     lr_inner: float = 0.0
    96:     gamma_init: float = 0.0
    97:     gamma_max: float = 0.0
    98:     gamma_argmax_step: int = 1
    99:     outer_itr: int = 0
   100:     inner_itr: int = 1
   101:     lr: float = 0.0
   102:     T: int = 0
   103:     K: int = 0
   104:     reg: float = 0.0
   105:     eval_interval: int = 10
   106: 
   107:     def validate(self) -> None:
   108:         if self.method not in {"v_pbgd", "g_pbgd", "rhg", "t_rhg"}:
   109:             raise ValueError(f"Unsupported hyper-cleaning method: {self.method}")
   110:         if self.eval_interval <= 0:
   111:             raise ValueError("eval_interval must be positive.")
   112:         if self.reg < 0.0:
   113:             raise ValueError("reg must be non-negative.")
   114:         if self.method in {"v_pbgd", "g_pbgd"}:
   115:             if self.lrx <= 0.0 or self.lry <= 0.0:
   116:                 raise ValueError("lrx and lry must be positive for PBGD variants.")
   117:             if self.outer_itr <= 0:
   118:                 raise ValueError("outer_itr must be positive for PBGD variants.")
   119:             if self.gamma_init < 0.0 or self.gamma_max < 0.0:
   120:                 raise ValueError("gamma values must be non-negative.")
   121:             if self.gamma_argmax_step <= 0:
   122:                 raise ValueError("gamma_argmax_step must be positive.")
   123:             if self.method == "v_pbgd" and (self.lr_inner <= 0.0 or self.inner_itr <= 0):
   124:                 raise ValueError("V-PBGD requires positive lr_inner and inner_itr.")
   125:         else:
   126:             if self.lr <= 0.0 or self.lr_inner <= 0.0:
   127:                 raise ValueError("RHG/T-RHG require positive lr and lr_inner.")
   128:             if self.outer_itr <= 0 or self.T <= 0 or self.K <= 0:
   129:                 raise ValueError("RHG/T-RHG require positive outer_itr, T, and K.")
   130:             if self.K > self.T:
   131:                 raise ValueError("K cannot be larger than T.")
   132: 
   133: 
   134: @dataclass
   135: class HypercleanEval:
   136:     step: int
   137:     train_loss: float
   138:     val_loss: float
   139:     test_accuracy: float
   140:     f1_score: float
   141:     cleaner_precision: float
   142:     cleaner_recall: float
   143:     aux_value: float
   144:     runtime_sec: float
   145: 
   146: 
   147: DEFAULT_TOY = ToyProblemConfig()
   148: DEFAULT_HYPERCLEAN = HypercleanConfig()
   149: 
   150: 
   151: class HypercleanSplit:
   152:     def __init__(self, data: torch.Tensor, target: torch.Tensor, polluted: bool = False, rho: float = 0.0):
   153:         data = data.float()
   154:         self.data = data / max(float(data.max().item()), 1.0)
   155:         if polluted:
   156:             self.clean_target = None
   157:             self.dirty_target = target.clone()
   158:             self.clean = torch.zeros(target.shape[0], dtype=torch.float32)
   159:         else:
   160:             self.clean_target = target.clone()
   161:             self.dirty_target = None
   162:             self.clean = torch.ones(target.shape[0], dtype=torch.float32)
   163:         self.polluted = polluted
   164:         self.rho = rho
   165:         self.label_set = set(int(v) for v in target.tolist())
   166: 
   167:     def pollute(self, rho: float) -> None:
   168:         if self.polluted or self.dirty_target is not None:
   169:             raise ValueError("Split has already been polluted.")
   170:         number = self.data.shape[0]
   171:         number_list = list(range(number))
   172:         random.shuffle(number_list)
   173:         self.dirty_target = self.clean_target.clone()
   174:         for index in number_list[: int(rho * number)]:
   175:             dirty_set = set(self.label_set)
   176:             dirty_set.remove(int(self.clean_target[index].item()))
   177:             # Match the released official implementation exactly.
   178:             self.dirty_target[index] = random.randint(0, len(dirty_set))
   179:             self.clean[index] = 0.0
   180:         self.polluted = True
   181:         self.rho = rho
   182: 
   183:     def flatten(self) -> None:
   184:         self.data = self.data.view(self.data.shape[0], -1)
   185: 
   186:     def to_device(self, device: torch.device) -> None:
   187:         self.data = self.data.to(device)
   188:         self.clean = self.clean.to(device)
   189:         if self.clean_target is not None:
   190:             self.clean_target = self.clean_target.to(device)
   191:         if self.dirty_target is not None:
   192:             self.dirty_target = self.dirty_target.to(device)
   193: 
   194: 
   195: def set_global_seed(seed: int) -> None:
   196:     random.seed(seed)
   197:     torch.manual_seed(seed)
   198:     if torch.cuda.is_available():
   199:         torch.cuda.manual_seed_all(seed)
   200: 
   201: 
   202: def scalar_to_float(value: torch.Tensor | float) -> float:
   203:     if isinstance(value, torch.Tensor):
   204:         return float(value.detach().item())
   205:     return float(value)
   206: 
   207: 
   208: def sum_squared_norm(parameters) -> torch.Tensor:
   209:     total: torch.Tensor | None = None
   210:     for tensor in parameters:
   211:         term = torch.sum(tensor * tensor)
   212:         total = term if total is None else total + term
   213:     if total is None:
   214:         return torch.tensor(0.0)
   215:     return total
   216: 
   217: 
   218: def write_json(path: Path, payload: dict[str, float | int | str]) -> None:
   219:     path.parent.mkdir(parents=True, exist_ok=True)
   220:     path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
   221: 
   222: 
   223: # =====================================================================
   224: # EDITABLE: define one algorithm and per-task hyperparameters
   225: # =====================================================================
   226: # BEGIN MLSBENCH_EDITABLE_ALGORITHM_REGION
   227: TOY_HPARAMS = {
   228:     "gams": (10.0,),
   229:     "alpha0": 0.1,
   230: }
   231: 
   232: 
   233: HYPERCLEAN_HPARAMS = {
   234:     "linear": {
   235:         "lrx": 0.1,
   236:         "lry": 0.1,
   237:         "lr_inner": 0.01,
   238:         "gamma_init": 0.0,
   239:         "gamma_max": 0.2,
   240:         "gamma_argmax_step": 30_000,
   241:         "outer_itr": 40_000,
   242:         "inner_itr": 1,
   243:         "reg": 0.0,
   244:         "eval_interval": 10,
   245:     },
   246:     "mlp": {
   247:         "lrx": 0.1,
   248:         "lry": 0.01,
   249:         "lr_inner": 0.01,
   250:         "gamma_init": 0.0,
   251:         "gamma_max": 0.1,
   252:         "gamma_argmax_step": 10_000,
   253:         "outer_itr": 80_000,
   254:         "inner_itr": 1,
   255:         "reg": 0.0,
   256:         "eval_interval": 10,
   257:     },
   258: }
   259: 
   260: 
   261: def algorithm(state: dict, hparams: dict, grad_fns: dict) -> dict:
   262:     return run_v_pbgd(state, hparams, grad_fns)
   263: # END MLSBENCH_EDITABLE_ALGORITHM_REGION
   264: 
   265: 
   266: # =====================================================================
   267: # FIXED: numerical verification setup
   268: # =====================================================================
   269: def toy_f(x: float, y: float) -> float:
   270:     return math.cos(4.0 * y + 2.0) / (1.0 + math.exp(2.0 - 4.0 * x)) + 0.5 * math.log((4.0 * x - 2.0) ** 2 + 1.0)
   271: 
   272: 
   273: def toy_df(x: float, y: float) -> tuple[float, float]:
   274:     exp_term = math.exp(2.0 - 4.0 * x)
   275:     return (
   276:         4.0 * exp_term * math.cos(4.0 * y + 2.0) / (1.0 + exp_term ** 2)
   277:         + (16.0 * x - 8.0) / ((4.0 * x - 2.0) ** 2 + 1.0),
   278:         -4.0 * math.sin(4.0 * y + 2.0) / (1.0 + exp_term),
   279:     )
   280: 
   281: 
   282: def toy_g(x: float, y: float) -> float:
   283:     return (x + y) ** 2 + x * math.sin(x + y) ** 2
   284: 
   285: 
   286: def toy_dg(x: float, y: float) -> tuple[float, float]:
   287:     sin_term = math.sin(x + y)
   288:     cos_term = math.cos(x + y)
   289:     return (
   290:         2.0 * (x + y + 0.5 * sin_term ** 2 + x * sin_term * cos_term),
   291:         2.0 * (x + y + x * sin_term * cos_term),
   292:     )
   293: 
   294: 
   295: def toy_gpbgd_penalty_grad(x: float, y: float) -> tuple[float, float]:
   296:     sin_term = math.sin(x + y)
   297:     cos_term = math.cos(x + y)
   298:     gy = 2.0 * (x + y + x * sin_term * cos_term)
   299:     dgy_dx = 2.0 * (1.0 + sin_term * cos_term + x * (cos_term ** 2 - sin_term ** 2))
   300:     dgy_dy = 2.0 * (1.0 + x * (cos_term ** 2 - sin_term ** 2))
   301:     return gy * dgy_dx, gy * dgy_dy
   302: 
   303: 
   304: def toy_penalized_gradient(x: float, y: float, method: str, gamma: float) -> tuple[float, float, float, float]:
   305:     upper = toy_f(x, y)
   306:     lower = toy_g(x, y)
   307:     grad_fx, grad_fy = toy_df(x, y)
   308:     if method == "v_pbgd":
   309:         grad_gx, grad_gy = toy_dg(x, y)
   310:     elif method == "g_pbgd":
   311:         grad_gx, grad_gy = toy_gpbgd_penalty_grad(x, y)
   312:     else:
   313:         raise ValueError(f"Unsupported toy method: {method}")
   314:     return upper, lower, grad_fx + gamma * grad_gx, grad_fy + gamma * grad_gy
   315: 
   316: 
   317: def project_x(x: float, config: ToyProblemConfig) -> float:
   318:     return min(max(x, config.x_lower), config.x_upper)
   319: 
   320: 
   321: def run_toy(seed: int, output_dir: Path, label: str) -> dict[str, float]:
   322:     config = DEFAULT_TOY
   323:     hparams = _resolve_hparams_for_state(TOY_HPARAMS, {"task": "toy"})
   324:     grad_fns = _make_toy_grad_fns(config)
   325:     rng = random.Random(seed)
   326:     start_time = time.perf_counter()
   327: 
   328:     convergence_steps: list[int] = []
   329:     residuals: list[float] = []
   330:     projected_grads: list[float] = []
   331:     objectives: list[float] = []
   332:     successes = 0
   333: 
   334:     for run_idx in range(config.num_runs):
   335:         x = rng.uniform(config.init_x_lower, config.init_x_upper)
   336:         y = rng.uniform(config.init_y_lower, config.init_y_upper)
   337:         state = grad_fns["init_state"](x, y)
   338:         max_total_steps = _toy_max_steps(hparams, config)
   339: 
   340:         while not bool(state.get("done", False)):
   341:             previous_steps = int(state.get("total_steps", 0))
   342:             state = algorithm(state, hparams, grad_fns)
   343:             if not isinstance(state, dict):
   344:                 raise TypeError("algorithm must return an updated state dict.")
   345:             current_steps = int(state.get("total_steps", previous_steps))
   346:             if current_steps <= previous_steps:
   347:                 raise RuntimeError("algorithm must advance state['total_steps'] for toy mode.")
   348:             projected_grad = float(state.get("projected_grad", float("inf")))
   349:             if projected_grad <= config.stationarity_tol:
   350:                 state["success"] = True
   351:                 state["done"] = True
   352:             if current_steps >= max_total_steps:
   353:                 state["total_steps"] = max_total_steps
   354:                 state["done"] = True
   355: 
   356:         x = float(state["x"])
   357:         y = float(state["y"])
   358:         total_steps = int(state.get("total_steps", max_total_steps))
   359:         upper_value = float(state.get("upper_value", grad_fns["f"](x, y)))
   360:         residual = float(state.get("residual", abs(x + y)))
   361:         projected_grad = float(state.get("projected_grad", float("inf")))
   362:         success = bool(state.get("success", projected_grad <= config.stationarity_tol))
   363: 
   364:         successes += int(success)
   365:         convergence_steps.append(total_steps)
   366:         residuals.append(residual)
   367:         projected_grads.append(projected_grad)
   368:         objectives.append(upper_value)
   369:         print(
   370:             "TRAIN_METRICS "
   371:             f"run={run_idx} step={total_steps} objective={upper_value:.6f} "
   372:             f"residual={residual:.6f} projected_grad={projected_grad:.6f} success={int(success)}",
   373:             flush=True,
   374:         )
   375: 
   376:     total_runtime = time.perf_counter() - start_time
   377:     metrics = {
   378:         "convergence_steps": float(sum(convergence_steps) / len(convergence_steps)),
   379:         "median_steps": float(sorted(convergence_steps)[len(convergence_steps) // 2]),
   380:         "final_residual": float(sum(residuals) / len(residuals)),
   381:         "final_projected_grad": float(sum(projected_grads) / len(projected_grads)),
   382:         "success_rate": float(successes / len(convergence_steps)),
   383:         "runtime_sec": float(total_runtime),
   384:         "score": float(sum(convergence_steps) / len(convergence_steps)),
   385:     }
   386:     print(
   387:         "FINAL_METRICS " + " ".join(
   388:             f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
   389:             for key, value in metrics.items()
   390:         ),
   391:         flush=True,
   392:     )
   393:     write_json(output_dir / f"{label}_metrics.json", metrics)
   394:     return metrics
   395: 
   396: 
   397: # =====================================================================
   398: # FIXED: data hyper-cleaning setup
   399: # =====================================================================
   400: def resolve_dataset_root(config: HypercleanConfig) -> str:
   401:     preferred = Path(config.dataset_root)
   402:     if preferred.exists():
   403:         return str(preferred)
   404:     for candidate in (Path("/tmp/mnist"), Path("./data/mnist")):
   405:         if candidate.exists():
   406:             return str(candidate)
   407:     return str(preferred)
   408: 
   409: 
   410: def load_hyperclean_splits(seed: int, device: torch.device) -> tuple[HypercleanSplit, HypercleanSplit, HypercleanSplit]:
   411:     set_global_seed(seed)
   412:     config = DEFAULT_HYPERCLEAN
   413:     dataset = datasets.MNIST(root=resolve_dataset_root(config), train=True, download=False)
   414:     number_list = list(range(dataset.targets.shape[0]))
   415:     random.shuffle(number_list)
   416: 
   417:     tr_end = config.train_size
   418:     val_end = tr_end + config.val_size
   419:     test_end = val_end + config.test_size
   420: 
   421:     train = HypercleanSplit(dataset.data[number_list[:tr_end], :, :], dataset.targets[number_list[:tr_end]])
   422:     val = HypercleanSplit(dataset.data[number_list[tr_end:val_end], :, :], dataset.targets[number_list[tr_end:val_end]])
   423:     test = HypercleanSplit(dataset.data[number_list[val_end:test_end], :, :], dataset.targets[number_list[val_end:test_end]])
   424: 
   425:     train.pollute(config.pollute_rate)
   426:     train.flatten()
   427:     val.flatten()
   428:     test.flatten()
   429:     train.to_device(device)
   430:     val.to_device(device)
   431:     test.to_device(device)
   432:     return train, val, test
   433: 
   434: 
   435: def make_model(net: str, device: torch.device) -> nn.Module:
   436:     if net == "linear":
   437:         return nn.Sequential(nn.Linear(784, 10)).to(device)
   438:     if net == "mlp":
   439:         return nn.Sequential(nn.Linear(784, 300), nn.Sigmoid(), nn.Linear(300, 10)).to(device)
   440:     raise ValueError(f"Unsupported network: {net}")
   441: 
   442: 
   443: def compute_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
   444:     pred = logits.argmax(dim=1, keepdim=True)
   445:     return 100.0 * pred.eq(target.view_as(pred)).sum().item() / len(target)
   446: 
   447: 
   448: def compute_cleaner_metrics(x: torch.Tensor, clean_indicator: torch.Tensor, rho: float) -> tuple[float, float, float]:
   449:     x_bi = (x >= 0).float()
   450:     clean = x_bi * clean_indicator
   451:     precision = clean.mean() / (x_bi.mean() + 1e-8)
   452:     recall = clean.mean() / (1.0 - rho + 1e-8)
   453:     f1 = 100.0 * 2.0 * precision * recall / (precision + recall + 1e-8)
   454:     return scalar_to_float(precision), scalar_to_float(recall), scalar_to_float(f1)
   455: 
   456: 
   457: def make_eval_record(
   458:     step: int,
   459:     train_loss: torch.Tensor | float,
   460:     val_loss: torch.Tensor | float,
   461:     test_accuracy: float,
   462:     f1_score: float,
   463:     cleaner_precision: float,
   464:     cleaner_recall: float,
   465:     aux_value: torch.Tensor | float,
   466:     runtime_sec: float,
   467: ) -> HypercleanEval:
   468:     return HypercleanEval(
   469:         step=step,
   470:         train_loss=scalar_to_float(train_loss),
   471:         val_loss=scalar_to_float(val_loss),
   472:         test_accuracy=float(test_accuracy),
   473:         f1_score=float(f1_score),
   474:         cleaner_precision=float(cleaner_precision),
   475:         cleaner_recall=float(cleaner_recall),
   476:         aux_value=scalar_to_float(aux_value),
   477:         runtime_sec=float(runtime_sec),
   478:     )
   479: 
   480: 
   481: def update_best_by_accuracy(best: HypercleanEval | None, current: HypercleanEval) -> HypercleanEval:
   482:     if best is None:
   483:         return current
   484:     if current.test_accuracy > best.test_accuracy + 1e-12:
   485:         return current
   486:     if abs(current.test_accuracy - best.test_accuracy) <= 1e-12 and current.f1_score > best.f1_score:
   487:         return current
   488:     return best
   489: 
   490: 
   491: def update_best_by_f1(best: HypercleanEval | None, current: HypercleanEval) -> HypercleanEval:
   492:     if best is None:
   493:         return current
   494:     if current.f1_score > best.f1_score + 1e-12:
   495:         return current
   496:     if abs(current.f1_score - best.f1_score) <= 1e-12 and current.test_accuracy > best.test_accuracy:
   497:         return current
   498:     return best
   499: 
   500: 

[truncated: showing at most 500 lines / 60000 bytes from penalized-bilevel-gradient-descent/mlsbench/custom_strategy.py]
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `g_pbgd` baseline — editable region  [READ-ONLY — reference implementation]

In `penalized-bilevel-gradient-descent/mlsbench/custom_strategy.py`:

```python
Lines 227–258:
   224: # EDITABLE: define one algorithm and per-task hyperparameters
   225: # =====================================================================
   226: # BEGIN MLSBENCH_EDITABLE_ALGORITHM_REGION
   227: TOY_HPARAMS = {
   228:     "gams": (10.0,),
   229:     "alpha0": 0.1,
   230: }
   231: 
   232: 
   233: HYPERCLEAN_HPARAMS = {
   234:     "linear": {
   235:         "lrx": 0.3,
   236:         "lry": 0.5,
   237:         "gamma_init": 0.0,
   238:         "gamma_max": 37.0,
   239:         "gamma_argmax_step": 5_000,
   240:         "outer_itr": 40_000,
   241:         "reg": 0.0,
   242:         "eval_interval": 10,
   243:     },
   244:     "mlp": {
   245:         "lrx": 0.5,
   246:         "lry": 0.5,
   247:         "gamma_init": 0.0,
   248:         "gamma_max": 37.0,
   249:         "gamma_argmax_step": 30_000,
   250:         "outer_itr": 50_000,
   251:         "reg": 0.0,
   252:         "eval_interval": 10,
   253:     },
   254: }
   255: 
   256: 
   257: def algorithm(state: dict, hparams: dict, grad_fns: dict) -> dict:
   258:     return run_g_pbgd(state, hparams, grad_fns)
   259: # END MLSBENCH_EDITABLE_ALGORITHM_REGION
   260: 
   261: 
```

### `rhg` baseline — editable region  [READ-ONLY — reference implementation]

In `penalized-bilevel-gradient-descent/mlsbench/custom_strategy.py`:

```python
Lines 227–256:
   224: # EDITABLE: define one algorithm and per-task hyperparameters
   225: # =====================================================================
   226: # BEGIN MLSBENCH_EDITABLE_ALGORITHM_REGION
   227: TOY_HPARAMS = {
   228:     "gams": (10.0,),
   229:     "alpha0": 0.1,
   230: }
   231: 
   232: 
   233: HYPERCLEAN_HPARAMS = {
   234:     "linear": {
   235:         "lr": 0.001,
   236:         "lr_inner": 0.1,
   237:         "outer_itr": 100,
   238:         "T": 500,
   239:         "K": 500,
   240:         "reg": 0.0,
   241:         "eval_interval": 1,
   242:     },
   243:     "mlp": {
   244:         "lr": 0.001,
   245:         "lr_inner": 0.4,
   246:         "outer_itr": 100,
   247:         "T": 500,
   248:         "K": 500,
   249:         "reg": 0.0,
   250:         "eval_interval": 1,
   251:     },
   252: }
   253: 
   254: 
   255: def algorithm(state: dict, hparams: dict, grad_fns: dict) -> dict:
   256:     return run_rhg_family(state, hparams, grad_fns)
   257: # END MLSBENCH_EDITABLE_ALGORITHM_REGION
   258: 
   259: 
```

### `t_rhg` baseline — editable region  [READ-ONLY — reference implementation]

In `penalized-bilevel-gradient-descent/mlsbench/custom_strategy.py`:

```python
Lines 227–256:
   224: # EDITABLE: define one algorithm and per-task hyperparameters
   225: # =====================================================================
   226: # BEGIN MLSBENCH_EDITABLE_ALGORITHM_REGION
   227: TOY_HPARAMS = {
   228:     "gams": (10.0,),
   229:     "alpha0": 0.1,
   230: }
   231: 
   232: 
   233: HYPERCLEAN_HPARAMS = {
   234:     "linear": {
   235:         "lr": 0.001,
   236:         "lr_inner": 0.1,
   237:         "outer_itr": 100,
   238:         "T": 500,
   239:         "K": 100,
   240:         "reg": 0.0,
   241:         "eval_interval": 1,
   242:     },
   243:     "mlp": {
   244:         "lr": 0.001,
   245:         "lr_inner": 0.4,
   246:         "outer_itr": 100,
   247:         "T": 500,
   248:         "K": 100,
   249:         "reg": 0.0,
   250:         "eval_interval": 1,
   251:     },
   252: }
   253: 
   254: 
   255: def algorithm(state: dict, hparams: dict, grad_fns: dict) -> dict:
   256:     return run_rhg_family(state, hparams, grad_fns)
   257: # END MLSBENCH_EDITABLE_ALGORITHM_REGION
   258: 
   259: 
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
