# MLS-Bench: optimization-parity

# Optimization Parity

## Research Question
Can you improve a fixed two-layer MLP's ability to learn sparse parity by designing only its initialization, training dataset, and AdamW hyperparameters?

## Background
The k-sparse parity problem maps a binary vector `x ∈ {0, 1}^N` to `y = (sum_{i in S} x_i) mod 2` for an unknown subset `S` of size `k = 8`. It is statistically easy but computationally hard (SQ-hard in `n^Ω(k)`), and it has become a canonical "feature-learning" benchmark. Barak, Edelman, Goel, Kakade, Malach, and Zhang, "Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit" (NeurIPS 2022; arXiv:2207.08799), show that vanilla SGD on a wide MLP undergoes a phase transition: the loss curve looks flat for a long time while a Fourier gap in the population gradient slowly amplifies, and only then does test accuracy jump.

In this benchmark the model architecture, optimizer family, batch size, training loop, and evaluation protocol are fixed. Your scientific freedom is in **initialization**, **training data construction**, and **AdamW hyperparameters** — the three knobs that prior work suggests can move the phase transition forward by orders of magnitude.

## What You Can Modify
Edit the scaffold file `pytorch-examples/optimization_parity/custom_strategy.py` only inside the editable block containing:

1. `init_model(model, config)`
2. `make_dataset(secret, config, seed)`
3. `get_optimizer_config(config)`

The benchmark is evaluated on three configurations: `(N=32, K=8)`, `(N=50, K=8)`, and `(N=64, K=8)`, all with `W = 512`.

## Fixed Setup
- Task: `y = (sum_{i in S} x_i) mod 2` for a hidden secret subset `S` of size `K = 8`.
- Inputs: binary vectors `x in {0, 1}^N`.
- Model: `Linear(N, W) -> ReLU -> Linear(W, 1) -> Sigmoid` with `W = 512`.
- Optimizer type: `AdamW`.
- Loss: binary cross-entropy.
- Batch size: 128.
- Training budget: up to 100,000 steps, reshuffling every epoch.
- Evaluation: 10 hidden secrets × 10 random epoch-orderings per secret = 100 runs; report mean held-out test accuracy.

## Interface Notes
- `init_model(...)` must not depend on the hidden secret.
- `make_dataset(...)` may use the provided secret and must return either `(x, y)` or `{"x": x, "y": y}`.
- `x` must have shape `[num_examples, N]` with binary values only.
- `y` must have shape `[num_examples]` (or `[num_examples, 1]`) with binary labels.
- Training dataset size must stay `<= 12_800_000` examples.
- `get_optimizer_config(...)` must return `lr`, `wd`, `beta1`, and `beta2`.

## Metric
The leaderboard metric is `test_accuracy` (also emitted as `score`), the mean test accuracy across all 100 training runs. Higher is better.

## Baselines (variants of the reference setup)
- **default** — single-pass training over freshly sampled examples with default AdamW settings (`lr = 1e-3`, `wd = 1e-2`, `(beta1, beta2) = (0.9, 0.999)`), the baseline analysed by Barak et al. (NeurIPS 2022; arXiv:2207.08799).
- **multi_epoch** — same configuration as `default` but iterating over a smaller fixed dataset for many epochs to test the impact of finite data and reshuffling.
- **nowd** — same as `default` but with `wd = 0`, isolating the role of weight decay during the slow-amplification phase identified in the paper.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/pytorch-examples/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `pytorch-examples/optimization_parity/custom_strategy.py`
- editable lines **220–255**




## Readable Context


### `pytorch-examples/optimization_parity/custom_strategy.py`  [EDITABLE — lines 220–255 only]

```python
     1: """Optimization-parity scaffold for MLS-Bench.
     2: 
     3: The fixed evaluation samples hidden sparse parity functions and asks the agent
     4: to control only:
     5:   1. model initialization
     6:   2. training-data generation
     7:   3. AdamW hyperparameters
     8: """
     9: 
    10: from __future__ import annotations
    11: 
    12: import argparse
    13: import json
    14: import math
    15: import random
    16: from dataclasses import asdict, dataclass, replace
    17: from pathlib import Path
    18: 
    19: import torch
    20: from torch import nn
    21: 
    22: 
    23: # =====================================================================
    24: # FIXED: Benchmark configuration
    25: # =====================================================================
    26: @dataclass(frozen=True)
    27: class TaskConfig:
    28:     n_features: int = 32
    29:     secret_size: int = 8
    30:     hidden_width: int = 512
    31:     batch_size: int = 128
    32:     max_steps: int = 30_000
    33:     max_train_examples: int = 12_800_000
    34:     num_hidden_secrets: int = 5
    35:     num_orderings: int = 3
    36:     test_set_size: int = 16_384
    37:     log_interval: int = 250
    38:     min_steps_before_stop: int = 1_000
    39:     early_stop_acc: float = 0.999
    40:     early_stop_windows: int = 4
    41: 
    42: 
    43: @dataclass(frozen=True)
    44: class OptimizerConfig:
    45:     lr: float
    46:     wd: float
    47:     beta1: float
    48:     beta2: float
    49: 
    50: 
    51: @dataclass(frozen=True)
    52: class RunResult:
    53:     secret_index: int
    54:     order_index: int
    55:     steps: int
    56:     test_accuracy: float
    57: 
    58: 
    59: DEFAULT_TASK = TaskConfig()
    60: 
    61: 
    62: def build_model(config: TaskConfig) -> nn.Sequential:
    63:     return nn.Sequential(
    64:         nn.Linear(config.n_features, config.hidden_width),
    65:         nn.ReLU(),
    66:         nn.Linear(config.hidden_width, 1),
    67:         nn.Sigmoid(),
    68:     )
    69: 
    70: 
    71: def set_global_seed(seed: int) -> None:
    72:     random.seed(seed)
    73:     torch.manual_seed(seed)
    74:     if torch.cuda.is_available():
    75:         torch.cuda.manual_seed_all(seed)
    76: 
    77: 
    78: def sample_hidden_secrets(config: TaskConfig, seed: int) -> list[tuple[int, ...]]:
    79:     max_unique = math.comb(config.n_features, config.secret_size)
    80:     if config.num_hidden_secrets > max_unique:
    81:         raise ValueError("Requested more hidden secrets than unique subsets.")
    82: 
    83:     rng = random.Random(seed)
    84:     seen: set[tuple[int, ...]] = set()
    85:     secrets: list[tuple[int, ...]] = []
    86:     while len(secrets) < config.num_hidden_secrets:
    87:         secret = tuple(sorted(rng.sample(range(config.n_features), config.secret_size)))
    88:         if secret not in seen:
    89:             seen.add(secret)
    90:             secrets.append(secret)
    91:     return secrets
    92: 
    93: 
    94: def parity_labels(x: torch.Tensor, secret: tuple[int, ...]) -> torch.Tensor:
    95:     secret_index = torch.tensor(secret, dtype=torch.long)
    96:     return (x.index_select(dim=1, index=secret_index).sum(dim=1).remainder(2)).to(
    97:         torch.float32
    98:     )
    99: 
   100: 
   101: def make_test_set(
   102:     secret: tuple[int, ...],
   103:     config: TaskConfig,
   104:     seed: int,
   105: ) -> tuple[torch.Tensor, torch.Tensor]:
   106:     generator = torch.Generator().manual_seed(seed)
   107:     x = torch.randint(
   108:         low=0,
   109:         high=2,
   110:         size=(config.test_set_size, config.n_features),
   111:         generator=generator,
   112:         dtype=torch.int64,
   113:     ).to(torch.float32)
   114:     y = parity_labels(x, secret)
   115:     return x, y
   116: 
   117: 
   118: def normalize_dataset(
   119:     dataset: object,
   120:     config: TaskConfig,
   121: ) -> tuple[torch.Tensor, torch.Tensor]:
   122:     if isinstance(dataset, dict):
   123:         if "x" not in dataset or "y" not in dataset:
   124:             raise ValueError("Dataset dict must contain 'x' and 'y'.")
   125:         x, y = dataset["x"], dataset["y"]
   126:     elif isinstance(dataset, (tuple, list)) and len(dataset) == 2:
   127:         x, y = dataset
   128:     else:
   129:         raise TypeError("Dataset must be a (x, y) pair or a dict with keys 'x' and 'y'.")
   130: 
   131:     x = torch.as_tensor(x, dtype=torch.float32)
   132:     y = torch.as_tensor(y, dtype=torch.float32).view(-1)
   133: 
   134:     if x.ndim != 2:
   135:         raise ValueError(f"Expected x to have shape [num_examples, n_features], got {tuple(x.shape)}.")
   136:     if x.shape[1] != config.n_features:
   137:         raise ValueError(
   138:             f"Expected x.shape[1] == {config.n_features}, got {x.shape[1]}."
   139:         )
   140:     if x.shape[0] != y.shape[0]:
   141:         raise ValueError("x and y must contain the same number of examples.")
   142:     if x.shape[0] == 0:
   143:         raise ValueError("Training dataset must contain at least one example.")
   144:     if x.shape[0] > config.max_train_examples:
   145:         raise ValueError(
   146:             f"Training dataset size {x.shape[0]} exceeds limit {config.max_train_examples}."
   147:         )
   148:     if not torch.all((x == 0) | (x == 1)):
   149:         raise ValueError("Training inputs must stay in {0, 1}.")
   150:     if not torch.all((y == 0) | (y == 1)):
   151:         raise ValueError("Training labels must stay in {0, 1}.")
   152:     return x.contiguous(), y.contiguous()
   153: 
   154: 
   155: def normalize_optimizer_config(config_dict: dict[str, float]) -> OptimizerConfig:
   156:     required = {"lr", "wd", "beta1", "beta2"}
   157:     missing = required - set(config_dict)
   158:     if missing:
   159:         raise ValueError(f"Missing optimizer hyperparameters: {sorted(missing)}")
   160: 
   161:     config = OptimizerConfig(
   162:         lr=float(config_dict["lr"]),
   163:         wd=float(config_dict["wd"]),
   164:         beta1=float(config_dict["beta1"]),
   165:         beta2=float(config_dict["beta2"]),
   166:     )
   167:     if not config.lr > 0.0:
   168:         raise ValueError("AdamW learning rate must be positive.")
   169:     if not config.wd >= 0.0:
   170:         raise ValueError("AdamW weight decay must be non-negative.")
   171:     if not 0.0 < config.beta1 < 1.0:
   172:         raise ValueError("AdamW beta1 must satisfy 0 < beta1 < 1.")
   173:     if not 0.0 < config.beta2 < 1.0:
   174:         raise ValueError("AdamW beta2 must satisfy 0 < beta2 < 1.")
   175:     return config
   176: 
   177: 
   178: def evaluate_accuracy(
   179:     model: nn.Module,
   180:     x: torch.Tensor,
   181:     y: torch.Tensor,
   182:     device: torch.device,
   183:     batch_size: int = 4096,
   184: ) -> float:
   185:     model.eval()
   186:     correct = 0
   187:     total = 0
   188:     with torch.no_grad():
   189:         for start in range(0, x.shape[0], batch_size):
   190:             end = start + batch_size
   191:             batch_x = x[start:end].to(device)
   192:             batch_y = y[start:end].to(device)
   193:             preds = model(batch_x).view(-1)
   194:             correct += ((preds >= 0.5) == (batch_y >= 0.5)).sum().item()
   195:             total += batch_y.numel()
   196:     return correct / max(total, 1)
   197: 
   198: 
   199: def maybe_log_final_window(
   200:     secret_index: int,
   201:     order_index: int,
   202:     steps: int,
   203:     window_loss: float,
   204:     window_acc: float,
   205:     window_count: int,
   206: ) -> None:
   207:     if window_count == 0:
   208:         return
   209:     print(
   210:         "TRAIN_METRICS "
   211:         f"secret={secret_index} order={order_index} step={steps} "
   212:         f"loss={window_loss / window_count:.6f} acc={window_acc / window_count:.6f}",
   213:         flush=True,
   214:     )
   215: 
   216: 
   217: # =====================================================================
   218: # EDITABLE: init_model, make_dataset, get_optimizer_config
   219: # =====================================================================
   220: def init_model(model: nn.Sequential, config: TaskConfig) -> None:
   221:     """Initialize the fixed two-layer MLP without using the hidden secret."""
   222:     for layer in model:
   223:         if isinstance(layer, nn.Linear):
   224:             gain = nn.init.calculate_gain("relu") if layer is model[0] else 1.0
   225:             nn.init.xavier_uniform_(layer.weight, gain=gain)
   226:             nn.init.zeros_(layer.bias)
   227: 
   228: 
   229: def make_dataset(
   230:     secret: tuple[int, ...],
   231:     config: TaskConfig,
   232:     seed: int,
   233: ) -> tuple[torch.Tensor, torch.Tensor]:
   234:     """Return a reproducible training dataset for one hidden secret."""
   235:     generator = torch.Generator().manual_seed(seed)
   236:     num_examples = 4_096
   237:     x = torch.randint(
   238:         low=0,
   239:         high=2,
   240:         size=(num_examples, config.n_features),
   241:         generator=generator,
   242:         dtype=torch.int64,
   243:     ).to(torch.float32)
   244:     y = parity_labels(x, secret)
   245:     return x, y
   246: 
   247: 
   248: def get_optimizer_config(config: TaskConfig) -> dict[str, float]:
   249:     """Return AdamW hyperparameters for the fixed training loop."""
   250:     return {
   251:         "lr": 1e-3,
   252:         "wd": 1e-2,
   253:         "beta1": 0.9,
   254:         "beta2": 0.999,
   255:     }
   256: 
   257: 
   258: # =====================================================================
   259: # FIXED: training and evaluation driver
   260: # =====================================================================
   261: def train_one_run(
   262:     train_x: torch.Tensor,
   263:     train_y: torch.Tensor,
   264:     test_x: torch.Tensor,
   265:     test_y: torch.Tensor,
   266:     config: TaskConfig,
   267:     device: torch.device,
   268:     run_seed: int,
   269:     order_seed: int,
   270:     secret_index: int,
   271:     order_index: int,
   272: ) -> RunResult:
   273:     set_global_seed(run_seed)
   274: 
   275:     model = build_model(config).to(device)
   276:     init_model(model, config)
   277:     optimizer_config = normalize_optimizer_config(get_optimizer_config(config))
   278:     optimizer = torch.optim.AdamW(
   279:         model.parameters(),
   280:         lr=optimizer_config.lr,
   281:         betas=(optimizer_config.beta1, optimizer_config.beta2),
   282:         weight_decay=optimizer_config.wd,
   283:     )
   284:     criterion = nn.BCELoss()
   285: 
   286:     steps = 0
   287:     stable_windows = 0
   288:     window_loss = 0.0
   289:     window_acc = 0.0
   290:     window_count = 0
   291:     last_logged_step = 0
   292:     permutation_generator = torch.Generator().manual_seed(order_seed)
   293: 
   294:     while steps < config.max_steps:
   295:         permutation = torch.randperm(train_x.shape[0], generator=permutation_generator)
   296:         for start in range(0, train_x.shape[0], config.batch_size):
   297:             batch_indices = permutation[start : start + config.batch_size]
   298:             batch_x = train_x.index_select(0, batch_indices).to(device)
   299:             batch_y = train_y.index_select(0, batch_indices).to(device)
   300: 
   301:             optimizer.zero_grad(set_to_none=True)
   302:             preds = model(batch_x).view(-1)
   303:             loss = criterion(preds, batch_y)
   304:             loss.backward()
   305:             optimizer.step()
   306: 
   307:             batch_acc = ((preds >= 0.5) == (batch_y >= 0.5)).float().mean().item()
   308:             window_loss += loss.item()
   309:             window_acc += batch_acc
   310:             window_count += 1
   311:             steps += 1
   312: 
   313:             should_log = steps == 1 or steps % config.log_interval == 0 or steps == config.max_steps
   314:             if should_log:
   315:                 avg_loss = window_loss / window_count
   316:                 avg_acc = window_acc / window_count
   317:                 print(
   318:                     "TRAIN_METRICS "
   319:                     f"secret={secret_index} order={order_index} step={steps} "
   320:                     f"loss={avg_loss:.6f} acc={avg_acc:.6f}",
   321:                     flush=True,
   322:                 )
   323:                 last_logged_step = steps
   324:                 if steps >= config.min_steps_before_stop and avg_acc >= config.early_stop_acc:
   325:                     stable_windows += 1
   326:                 else:
   327:                     stable_windows = 0
   328:                 window_loss = 0.0
   329:                 window_acc = 0.0
   330:                 window_count = 0
   331:                 if stable_windows >= config.early_stop_windows:
   332:                     break
   333: 
   334:             if steps >= config.max_steps:
   335:                 break
   336:         if stable_windows >= config.early_stop_windows or steps >= config.max_steps:
   337:             break
   338: 
   339:     if last_logged_step != steps:
   340:         maybe_log_final_window(
   341:             secret_index=secret_index,
   342:             order_index=order_index,
   343:             steps=steps,
   344:             window_loss=window_loss,
   345:             window_acc=window_acc,
   346:             window_count=window_count,
   347:         )
   348: 
   349:     test_accuracy = evaluate_accuracy(model, test_x, test_y, device)
   350:     print(
   351:         "RUN_METRICS "
   352:         f"secret={secret_index} order={order_index} steps={steps} "
   353:         f"test_accuracy={test_accuracy:.6f}",
   354:         flush=True,
   355:     )
   356:     return RunResult(
   357:         secret_index=secret_index,
   358:         order_index=order_index,
   359:         steps=steps,
   360:         test_accuracy=test_accuracy,
   361:     )
   362: 
   363: 
   364: def resolve_device(device_arg: str) -> torch.device:
   365:     if device_arg == "cpu":
   366:         return torch.device("cpu")
   367:     if device_arg == "cuda":
   368:         if not torch.cuda.is_available():
   369:             raise RuntimeError("CUDA requested but no GPU is available.")
   370:         return torch.device("cuda")
   371:     return torch.device("cuda" if torch.cuda.is_available() else "cpu")
   372: 
   373: 
   374: def maybe_apply_smoke_mode(config: TaskConfig, enabled: bool) -> TaskConfig:
   375:     if not enabled:
   376:         return config
   377:     return replace(
   378:         config,
   379:         num_hidden_secrets=2,
   380:         num_orderings=2,
   381:         test_set_size=2_048,
   382:         max_steps=4_000,
   383:         log_interval=100,
   384:         min_steps_before_stop=400,
   385:         early_stop_windows=3,
   386:     )
   387: 
   388: 
   389: def run_benchmark(
   390:     config: TaskConfig,
   391:     seed: int,
   392:     device: torch.device,
   393: ) -> dict[str, object]:
   394:     print(
   395:         "TASK_CONFIG "
   396:         + " ".join(
   397:             [
   398:                 f"N={config.n_features}",
   399:                 f"K={config.secret_size}",
   400:                 f"W={config.hidden_width}",
   401:                 f"num_hidden_secrets={config.num_hidden_secrets}",
   402:                 f"num_orderings={config.num_orderings}",
   403:                 f"test_set_size={config.test_set_size}",
   404:                 f"batch_size={config.batch_size}",
   405:                 f"max_steps={config.max_steps}",
   406:             ]
   407:         ),
   408:         flush=True,
   409:     )
   410: 
   411:     secrets = sample_hidden_secrets(config, seed + 17)
   412:     results: list[RunResult] = []
   413: 
   414:     for secret_index, secret in enumerate(secrets):
   415:         train_dataset_seed = seed * 10_000 + secret_index
   416:         train_x, train_y = normalize_dataset(
   417:             make_dataset(secret, config, train_dataset_seed),
   418:             config,
   419:         )
   420:         test_x, test_y = make_test_set(
   421:             secret=secret,
   422:             config=config,
   423:             seed=seed * 20_000 + secret_index,
   424:         )
   425:         positive_rate = float(train_y.mean().item())
   426:         print(
   427:             "DATASET_METRICS "
   428:             f"secret={secret_index} num_examples={train_x.shape[0]} "
   429:             f"positive_rate={positive_rate:.6f}",
   430:             flush=True,
   431:         )
   432: 
   433:         for order_index in range(config.num_orderings):
   434:             run_seed = seed * 1_000_000 + secret_index * 1_000 + order_index
   435:             order_seed = seed * 2_000_000 + secret_index * 1_000 + order_index
   436:             results.append(
   437:                 train_one_run(
   438:                     train_x=train_x,
   439:                     train_y=train_y,
   440:                     test_x=test_x,
   441:                     test_y=test_y,
   442:                     config=config,
   443:                     device=device,
   444:                     run_seed=run_seed,
   445:                     order_seed=order_seed,
   446:                     secret_index=secret_index,
   447:                     order_index=order_index,
   448:                 )
   449:             )
   450: 
   451:     accuracy_tensor = torch.tensor([result.test_accuracy for result in results], dtype=torch.float64)
   452:     step_tensor = torch.tensor([result.steps for result in results], dtype=torch.float64)
   453:     final_metrics = {
   454:         "test_accuracy": float(accuracy_tensor.mean().item()),
   455:         "score": float(accuracy_tensor.mean().item()),
   456:         "test_accuracy_std": float(accuracy_tensor.std(unbiased=False).item()),
   457:         "mean_steps": float(step_tensor.mean().item()),
   458:         "num_runs": int(len(results)),
   459:     }
   460:     print(
   461:         "FINAL_METRICS "
   462:         + " ".join(
   463:             f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
   464:             for key, value in final_metrics.items()
   465:         ),
   466:         flush=True,
   467:     )
   468:     # Also print TEST_METRICS for framework compatibility
   469:     print(
   470:         f"TEST_METRICS test_accuracy={final_metrics['test_accuracy']:.6f} "
   471:         f"score={final_metrics['score']:.6f}",
   472:         flush=True,
   473:     )
   474:     return {
   475:         "config": asdict(config),
   476:         "metrics": final_metrics,
   477:         "results": [asdict(result) for result in results],
   478:     }
   479: 
   480: 
   481: def parse_args() -> argparse.Namespace:
   482:     parser = argparse.ArgumentParser(description="Run the MLS-Bench optimization-parity task.")
   483:     parser.add_argument("--seed", type=int, default=42, help="Top-level benchmark seed.")
   484:     parser.add_argument(
   485:         "--output-dir",
   486:         type=Path,
   487:         default=None,
   488:         help="Optional directory for a JSON summary.",
   489:     )
   490:     parser.add_argument(
   491:         "--label",
   492:         type=str,
   493:         default="eval",
   494:         help="Optional label stored in the JSON summary.",
   495:     )
   496:     parser.add_argument(
   497:         "--device",
   498:         choices=("auto", "cpu", "cuda"),
   499:         default="auto",
   500:         help="Execution device.",

[truncated: showing at most 500 lines / 60000 bytes from pytorch-examples/optimization_parity/custom_strategy.py]
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **n32-k8** — wall-clock budget `0:59:00`, compute share `1.0`
- **n50-k8** — wall-clock budget `0:59:00`, compute share `1.0`
- **n64-k8** — wall-clock budget `0:59:00`, compute share `1.0`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `default` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-examples/optimization_parity/custom_strategy.py`:

```python
Lines 220–255:
   217: # =====================================================================
   218: # EDITABLE: init_model, make_dataset, get_optimizer_config
   219: # =====================================================================
   220: def init_model(model: nn.Sequential, config: TaskConfig) -> None:
   221:     """Initialize the fixed two-layer MLP without using the hidden secret."""
   222:     for layer in model:
   223:         if isinstance(layer, nn.Linear):
   224:             gain = nn.init.calculate_gain("relu") if layer is model[0] else 1.0
   225:             nn.init.xavier_uniform_(layer.weight, gain=gain)
   226:             nn.init.zeros_(layer.bias)
   227: 
   228: 
   229: def make_dataset(
   230:     secret: tuple[int, ...],
   231:     config: TaskConfig,
   232:     seed: int,
   233: ) -> tuple[torch.Tensor, torch.Tensor]:
   234:     """Return a maximal random dataset to induce one-pass training."""
   235:     generator = torch.Generator().manual_seed(seed)
   236:     num_examples = config.max_train_examples
   237:     x = torch.randint(
   238:         low=0,
   239:         high=2,
   240:         size=(num_examples, config.n_features),
   241:         generator=generator,
   242:         dtype=torch.int64,
   243:     ).to(torch.float32)
   244:     y = parity_labels(x, secret)
   245:     return x, y
   246: 
   247: 
   248: def get_optimizer_config(config: TaskConfig) -> dict[str, float]:
   249:     """Return AdamW hyperparameters for the fixed training loop."""
   250:     return {
   251:         "lr": 1e-3,
   252:         "wd": 1e-2,
   253:         "beta1": 0.9,
   254:         "beta2": 0.999,
   255:     }
   256: 
   257: 
   258: # =====================================================================
```

### `multi_epoch` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-examples/optimization_parity/custom_strategy.py`:

```python
Lines 220–257:
   217: # =====================================================================
   218: # EDITABLE: init_model, make_dataset, get_optimizer_config
   219: # =====================================================================
   220: def init_model(model: nn.Sequential, config: TaskConfig) -> None:
   221:     """Initialize the fixed two-layer MLP without using the hidden secret."""
   222:     for layer in model:
   223:         if isinstance(layer, nn.Linear):
   224:             gain = nn.init.calculate_gain("relu") if layer is model[0] else 1.0
   225:             nn.init.xavier_uniform_(layer.weight, gain=gain)
   226:             nn.init.zeros_(layer.bias)
   227: 
   228: 
   229: def make_dataset(
   230:     secret: tuple[int, ...],
   231:     config: TaskConfig,
   232:     seed: int,
   233: ) -> tuple[torch.Tensor, torch.Tensor]:
   234:     """Use a smaller, configurable random dataset to allow multi-epoch reuse."""
   235:     generator = torch.Generator().manual_seed(seed)
   236:     train_examples = 10_000  # Tunable parameter for this multi-epoch baseline.
   237:     num_examples = min(train_examples, config.max_train_examples)
   238: 
   239:     x = torch.randint(
   240:         low=0,
   241:         high=2,
   242:         size=(num_examples, config.n_features),
   243:         generator=generator,
   244:         dtype=torch.int64,
   245:     ).to(torch.float32)
   246:     y = parity_labels(x, secret)
   247:     return x, y
   248: 
   249: 
   250: def get_optimizer_config(config: TaskConfig) -> dict[str, float]:
   251:     """Return AdamW hyperparameters for the fixed training loop."""
   252:     return {
   253:         "lr": 1e-3,
   254:         "wd": 1e-2,
   255:         "beta1": 0.9,
   256:         "beta2": 0.999,
   257:     }
   258: 
   259: 
   260: # =====================================================================
```

### `nowd` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-examples/optimization_parity/custom_strategy.py`:

```python
Lines 220–255:
   217: # =====================================================================
   218: # EDITABLE: init_model, make_dataset, get_optimizer_config
   219: # =====================================================================
   220: def init_model(model: nn.Sequential, config: TaskConfig) -> None:
   221:     """Initialize the fixed two-layer MLP without using the hidden secret."""
   222:     for layer in model:
   223:         if isinstance(layer, nn.Linear):
   224:             gain = nn.init.calculate_gain("relu") if layer is model[0] else 1.0
   225:             nn.init.xavier_uniform_(layer.weight, gain=gain)
   226:             nn.init.zeros_(layer.bias)
   227: 
   228: 
   229: def make_dataset(
   230:     secret: tuple[int, ...],
   231:     config: TaskConfig,
   232:     seed: int,
   233: ) -> tuple[torch.Tensor, torch.Tensor]:
   234:     """Return a maximal random dataset to induce one-pass training."""
   235:     generator = torch.Generator().manual_seed(seed)
   236:     num_examples = config.max_train_examples
   237:     x = torch.randint(
   238:         low=0,
   239:         high=2,
   240:         size=(num_examples, config.n_features),
   241:         generator=generator,
   242:         dtype=torch.int64,
   243:     ).to(torch.float32)
   244:     y = parity_labels(x, secret)
   245:     return x, y
   246: 
   247: 
   248: def get_optimizer_config(config: TaskConfig) -> dict[str, float]:
   249:     """Return AdamW hyperparameters with no weight decay."""
   250:     return {
   251:         "lr": 1e-3,
   252:         "wd": 0.0,
   253:         "beta1": 0.9,
   254:         "beta2": 0.999,
   255:     }
   256: 
   257: 
   258: # =====================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
