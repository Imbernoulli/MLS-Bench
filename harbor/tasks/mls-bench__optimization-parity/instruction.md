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

## Fixed Setup
- Task: `y = (sum_{i in S} x_i) mod 2` for a hidden secret subset `S`.
- Inputs: binary vectors `x in {0, 1}^N`.
- The model architecture, optimizer family (`AdamW`), loss, batch size, training loop, and evaluation protocol are fixed by the harness and not editable. All sizes are provided to your functions via the `config` (`TaskConfig`) object.

## Interface Notes
- `init_model(...)` must not depend on the hidden secret.
- `make_dataset(...)` may use the provided secret and must return either `(x, y)` or `{"x": x, "y": y}`.
- `x` must have shape `[num_examples, N]` with binary values only.
- `y` must have shape `[num_examples]` (or `[num_examples, 1]`) with binary labels.
- Training dataset size must stay `<= 12_800_000` examples.
- `get_optimizer_config(...)` must return `lr`, `wd`, `beta1`, and `beta2`.

## Baselines (variants of the reference setup)
- **default** — single-pass training over freshly sampled examples with default AdamW settings (`lr = 1e-3`, `wd = 1e-2`, `(beta1, beta2) = (0.9, 0.999)`), the baseline analysed by Barak et al. (NeurIPS 2022; arXiv:2207.08799).
- **multi_epoch** — same configuration as `default` but iterating over a smaller fixed dataset for many epochs to test the impact of finite data and reshuffling.
- **nowd** — same as `default` but with `wd = 0`, isolating the role of weight decay during the slow-amplification phase identified in the paper.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/pytorch-examples/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits that change code outside these ranges — or creating new files, or
deleting whole files — will cause your submission to be invalid.

The line numbers mark an editable **region**, not a fixed line-count budget: you
may add or remove lines inside it. Only code outside the editable ranges must
stay unchanged.

- `pytorch-examples/optimization_parity/custom_strategy.py`
- editable lines **242–274**




## Readable Context


### `pytorch-examples/optimization_parity/custom_strategy.py`  [EDITABLE — lines 242–274 only]

```python
     1: """Optimization-parity scaffold for MLS-Bench.
     2: 
     3: The fixed evaluation trains a fixed two-layer MLP to learn hidden sparse parity
     4: functions and asks the agent to control only:
     5:   1. model initialization
     6:   2. training-data construction (selecting/transforming the provided pool)
     7:   3. AdamW hyperparameters
     8: 
     9: NOTE: The hidden parity secret S and the held-out test labels are NOT part of
    10: this program. The harness pre-generates the (unlabeled) training inputs and
    11: their labels for you and scores your predictions against held-out truth in a
    12: separate host-side process. Your editable hooks only ever see binary inputs and
    13: their labels — never the secret subset S, and never the test labels. The runner
    14: trains your model and emits its predictions on a held-out test set; the host
    15: regenerates the test labels and computes test accuracy.
    16: """
    17: 
    18: from __future__ import annotations
    19: 
    20: import argparse
    21: import base64
    22: import json
    23: import os
    24: import random
    25: from dataclasses import asdict, dataclass, replace
    26: from pathlib import Path
    27: 
    28: import torch
    29: from torch import nn
    30: 
    31: 
    32: # =====================================================================
    33: # FIXED: Benchmark configuration
    34: # =====================================================================
    35: @dataclass(frozen=True)
    36: class TaskConfig:
    37:     n_features: int = 32
    38:     secret_size: int = 8
    39:     hidden_width: int = 512
    40:     batch_size: int = 128
    41:     max_steps: int = 30_000
    42:     max_train_examples: int = 12_800_000
    43:     num_hidden_secrets: int = 5
    44:     num_orderings: int = 3
    45:     test_set_size: int = 16_384
    46:     log_interval: int = 250
    47:     min_steps_before_stop: int = 1_000
    48:     early_stop_acc: float = 0.999
    49:     early_stop_windows: int = 4
    50: 
    51: 
    52: @dataclass(frozen=True)
    53: class OptimizerConfig:
    54:     lr: float
    55:     wd: float
    56:     beta1: float
    57:     beta2: float
    58: 
    59: 
    60: @dataclass(frozen=True)
    61: class RunResult:
    62:     secret_index: int
    63:     order_index: int
    64:     steps: int
    65: 
    66: 
    67: DEFAULT_TASK = TaskConfig()
    68: 
    69: 
    70: def build_model(config: TaskConfig) -> nn.Sequential:
    71:     return nn.Sequential(
    72:         nn.Linear(config.n_features, config.hidden_width),
    73:         nn.ReLU(),
    74:         nn.Linear(config.hidden_width, 1),
    75:         nn.Sigmoid(),
    76:     )
    77: 
    78: 
    79: def set_global_seed(seed: int) -> None:
    80:     random.seed(seed)
    81:     torch.manual_seed(seed)
    82:     if torch.cuda.is_available():
    83:         torch.cuda.manual_seed_all(seed)
    84: 
    85: 
    86: def normalize_dataset(
    87:     dataset: object,
    88:     config: TaskConfig,
    89: ) -> tuple[torch.Tensor, torch.Tensor]:
    90:     if isinstance(dataset, dict):
    91:         if "x" not in dataset or "y" not in dataset:
    92:             raise ValueError("Dataset dict must contain 'x' and 'y'.")
    93:         x, y = dataset["x"], dataset["y"]
    94:     elif isinstance(dataset, (tuple, list)) and len(dataset) == 2:
    95:         x, y = dataset
    96:     else:
    97:         raise TypeError("Dataset must be a (x, y) pair or a dict with keys 'x' and 'y'.")
    98: 
    99:     x = torch.as_tensor(x, dtype=torch.float32)
   100:     y = torch.as_tensor(y, dtype=torch.float32).view(-1)
   101: 
   102:     if x.ndim != 2:
   103:         raise ValueError(f"Expected x to have shape [num_examples, n_features], got {tuple(x.shape)}.")
   104:     if x.shape[1] != config.n_features:
   105:         raise ValueError(
   106:             f"Expected x.shape[1] == {config.n_features}, got {x.shape[1]}."
   107:         )
   108:     if x.shape[0] != y.shape[0]:
   109:         raise ValueError("x and y must contain the same number of examples.")
   110:     if x.shape[0] == 0:
   111:         raise ValueError("Training dataset must contain at least one example.")
   112:     if x.shape[0] > config.max_train_examples:
   113:         raise ValueError(
   114:             f"Training dataset size {x.shape[0]} exceeds limit {config.max_train_examples}."
   115:         )
   116:     if not torch.all((x == 0) | (x == 1)):
   117:         raise ValueError("Training inputs must stay in {0, 1}.")
   118:     if not torch.all((y == 0) | (y == 1)):
   119:         raise ValueError("Training labels must stay in {0, 1}.")
   120:     return x.contiguous(), y.contiguous()
   121: 
   122: 
   123: def normalize_optimizer_config(config_dict: dict[str, float]) -> OptimizerConfig:
   124:     required = {"lr", "wd", "beta1", "beta2"}
   125:     missing = required - set(config_dict)
   126:     if missing:
   127:         raise ValueError(f"Missing optimizer hyperparameters: {sorted(missing)}")
   128: 
   129:     config = OptimizerConfig(
   130:         lr=float(config_dict["lr"]),
   131:         wd=float(config_dict["wd"]),
   132:         beta1=float(config_dict["beta1"]),
   133:         beta2=float(config_dict["beta2"]),
   134:     )
   135:     if not config.lr > 0.0:
   136:         raise ValueError("AdamW learning rate must be positive.")
   137:     if not config.wd >= 0.0:
   138:         raise ValueError("AdamW weight decay must be non-negative.")
   139:     if not 0.0 < config.beta1 < 1.0:
   140:         raise ValueError("AdamW beta1 must satisfy 0 < beta1 < 1.")
   141:     if not 0.0 < config.beta2 < 1.0:
   142:         raise ValueError("AdamW beta2 must satisfy 0 < beta2 < 1.")
   143:     return config
   144: 
   145: 
   146: def predict_on(
   147:     model: nn.Module,
   148:     x: torch.Tensor,
   149:     device: torch.device,
   150:     batch_size: int = 4096,
   151: ) -> torch.Tensor:
   152:     """Return raw model outputs (sigmoid probabilities) for every row of x."""
   153:     model.eval()
   154:     outputs = []
   155:     with torch.no_grad():
   156:         for start in range(0, x.shape[0], batch_size):
   157:             end = start + batch_size
   158:             batch_x = x[start:end].to(device)
   159:             preds = model(batch_x).view(-1)
   160:             outputs.append(preds.detach().cpu())
   161:     return torch.cat(outputs) if outputs else torch.empty(0)
   162: 
   163: 
   164: def maybe_log_final_window(
   165:     secret_index: int,
   166:     order_index: int,
   167:     steps: int,
   168:     window_loss: float,
   169:     window_acc: float,
   170:     window_count: int,
   171: ) -> None:
   172:     if window_count == 0:
   173:         return
   174:     print(
   175:         "TRAIN_METRICS "
   176:         f"secret={secret_index} order={order_index} step={steps} "
   177:         f"loss={window_loss / window_count:.6f} acc={window_acc / window_count:.6f}",
   178:         flush=True,
   179:     )
   180: 
   181: 
   182: # =====================================================================
   183: # FIXED: held-out input loading (the harness pre-generates these; the
   184: # secret and the test labels are never present in this process)
   185: # =====================================================================
   186: def _inputs_dir() -> str:
   187:     """Directory holding the pre-generated parity inputs for this task."""
   188:     env = os.environ.get("PARITY_INPUTS_DIR")
   189:     if env:
   190:         return env
   191:     return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_parity_inputs")
   192: 
   193: 
   194: def _config_tag(config: TaskConfig) -> str:
   195:     return f"n{config.n_features}_k{config.secret_size}"
   196: 
   197: 
   198: def gen_train_pool_x(config: TaskConfig, train_dataset_seed: int) -> torch.Tensor:
   199:     """Regenerate the full unlabeled training x-pool (no secret involved)."""
   200:     generator = torch.Generator().manual_seed(train_dataset_seed)
   201:     return torch.randint(
   202:         low=0,
   203:         high=2,
   204:         size=(config.max_train_examples, config.n_features),
   205:         generator=generator,
   206:         dtype=torch.int64,
   207:     ).to(torch.float32)
   208: 
   209: 
   210: def gen_test_x(config: TaskConfig, test_seed: int) -> torch.Tensor:
   211:     """Regenerate the held-out test inputs (no secret; labels are withheld)."""
   212:     generator = torch.Generator().manual_seed(test_seed)
   213:     return torch.randint(
   214:         low=0,
   215:         high=2,
   216:         size=(config.test_set_size, config.n_features),
   217:         generator=generator,
   218:         dtype=torch.int64,
   219:     ).to(torch.float32)
   220: 
   221: 
   222: def load_train_labels(config: TaskConfig, seed: int, secret_index: int) -> torch.Tensor:
   223:     """Load the bit-packed training-pool labels for one hidden secret.
   224: 
   225:     Only the labels are provided (the secret that produced them is held out).
   226:     The labels are bit-packed for one row per training example over the full
   227:     ``max_train_examples`` pool; unpack to a float tensor in {0, 1}.
   228:     """
   229:     import numpy as np
   230: 
   231:     tag = _config_tag(config)
   232:     path = os.path.join(_inputs_dir(), f"{tag}_seed{seed}_s{secret_index}.labels.b64")
   233:     with open(path, "r") as f:
   234:         packed = np.frombuffer(base64.b64decode(f.read()), dtype=np.uint8)
   235:     bits = np.unpackbits(packed)[: config.max_train_examples]
   236:     return torch.from_numpy(bits.astype("float32"))
   237: 
   238: 
   239: # =====================================================================
   240: # EDITABLE: init_model, make_dataset, get_optimizer_config
   241: # =====================================================================
   242: def init_model(model: nn.Sequential, config: TaskConfig) -> None:
   243:     """Initialize the fixed two-layer MLP."""
   244:     for layer in model:
   245:         if isinstance(layer, nn.Linear):
   246:             gain = nn.init.calculate_gain("relu") if layer is model[0] else 1.0
   247:             nn.init.xavier_uniform_(layer.weight, gain=gain)
   248:             nn.init.zeros_(layer.bias)
   249: 
   250: 
   251: def make_dataset(
   252:     x_pool: torch.Tensor,
   253:     y_pool: torch.Tensor,
   254:     config: TaskConfig,
   255: ) -> tuple[torch.Tensor, torch.Tensor]:
   256:     """Construct the training dataset from the provided labeled pool.
   257: 
   258:     The harness supplies a large pool of labeled binary examples (x_pool, y_pool)
   259:     drawn from the same distribution as the held-out test set. Select and/or
   260:     transform it however you like; the result must be a binary (x, y) pair (or a
   261:     dict with keys 'x' and 'y'). The hidden parity secret is never exposed.
   262:     """
   263:     num_examples = 4_096
   264:     return x_pool[:num_examples], y_pool[:num_examples]
   265: 
   266: 
   267: def get_optimizer_config(config: TaskConfig) -> dict[str, float]:
   268:     """Return AdamW hyperparameters for the fixed training loop."""
   269:     return {
   270:         "lr": 1e-3,
   271:         "wd": 1e-2,
   272:         "beta1": 0.9,
   273:         "beta2": 0.999,
   274:     }
   275: 
   276: 
   277: # =====================================================================
   278: # FIXED: training and prediction driver
   279: # =====================================================================
   280: def train_one_run(
   281:     train_x: torch.Tensor,
   282:     train_y: torch.Tensor,
   283:     test_x: torch.Tensor,
   284:     config: TaskConfig,
   285:     device: torch.device,
   286:     run_seed: int,
   287:     order_seed: int,
   288:     secret_index: int,
   289:     order_index: int,
   290: ) -> tuple[RunResult, torch.Tensor]:
   291:     set_global_seed(run_seed)
   292: 
   293:     model = build_model(config).to(device)
   294:     init_model(model, config)
   295:     optimizer_config = normalize_optimizer_config(get_optimizer_config(config))
   296:     optimizer = torch.optim.AdamW(
   297:         model.parameters(),
   298:         lr=optimizer_config.lr,
   299:         betas=(optimizer_config.beta1, optimizer_config.beta2),
   300:         weight_decay=optimizer_config.wd,
   301:     )
   302:     criterion = nn.BCELoss()
   303: 
   304:     steps = 0
   305:     stable_windows = 0
   306:     window_loss = 0.0
   307:     window_acc = 0.0
   308:     window_count = 0
   309:     last_logged_step = 0
   310:     permutation_generator = torch.Generator().manual_seed(order_seed)
   311: 
   312:     while steps < config.max_steps:
   313:         permutation = torch.randperm(train_x.shape[0], generator=permutation_generator)
   314:         for start in range(0, train_x.shape[0], config.batch_size):
   315:             batch_indices = permutation[start : start + config.batch_size]
   316:             batch_x = train_x.index_select(0, batch_indices).to(device)
   317:             batch_y = train_y.index_select(0, batch_indices).to(device)
   318: 
   319:             optimizer.zero_grad(set_to_none=True)
   320:             preds = model(batch_x).view(-1)
   321:             loss = criterion(preds, batch_y)
   322:             loss.backward()
   323:             optimizer.step()
   324: 
   325:             batch_acc = ((preds >= 0.5) == (batch_y >= 0.5)).float().mean().item()
   326:             window_loss += loss.item()
   327:             window_acc += batch_acc
   328:             window_count += 1
   329:             steps += 1
   330: 
   331:             should_log = steps == 1 or steps % config.log_interval == 0 or steps == config.max_steps
   332:             if should_log:
   333:                 avg_loss = window_loss / window_count
   334:                 avg_acc = window_acc / window_count
   335:                 print(
   336:                     "TRAIN_METRICS "
   337:                     f"secret={secret_index} order={order_index} step={steps} "
   338:                     f"loss={avg_loss:.6f} acc={avg_acc:.6f}",
   339:                     flush=True,
   340:                 )
   341:                 last_logged_step = steps
   342:                 if steps >= config.min_steps_before_stop and avg_acc >= config.early_stop_acc:
   343:                     stable_windows += 1
   344:                 else:
   345:                     stable_windows = 0
   346:                 window_loss = 0.0
   347:                 window_acc = 0.0
   348:                 window_count = 0
   349:                 if stable_windows >= config.early_stop_windows:
   350:                     break
   351: 
   352:             if steps >= config.max_steps:
   353:                 break
   354:         if stable_windows >= config.early_stop_windows or steps >= config.max_steps:
   355:             break
   356: 
   357:     if last_logged_step != steps:
   358:         maybe_log_final_window(
   359:             secret_index=secret_index,
   360:             order_index=order_index,
   361:             steps=steps,
   362:             window_loss=window_loss,
   363:             window_acc=window_acc,
   364:             window_count=window_count,
   365:         )
   366: 
   367:     test_preds = predict_on(model, test_x, device)
   368:     print(
   369:         "RUN_METRICS "
   370:         f"secret={secret_index} order={order_index} steps={steps}",
   371:         flush=True,
   372:     )
   373:     return (
   374:         RunResult(
   375:             secret_index=secret_index,
   376:             order_index=order_index,
   377:             steps=steps,
   378:         ),
   379:         test_preds,
   380:     )
   381: 
   382: 
   383: def resolve_device(device_arg: str) -> torch.device:
   384:     if device_arg == "cpu":
   385:         return torch.device("cpu")
   386:     if device_arg == "cuda":
   387:         if not torch.cuda.is_available():
   388:             raise RuntimeError("CUDA requested but no GPU is available.")
   389:         return torch.device("cuda")
   390:     return torch.device("cuda" if torch.cuda.is_available() else "cpu")
   391: 
   392: 
   393: def maybe_apply_smoke_mode(config: TaskConfig, enabled: bool) -> TaskConfig:
   394:     if not enabled:
   395:         return config
   396:     return replace(
   397:         config,
   398:         num_hidden_secrets=2,
   399:         num_orderings=2,
   400:         test_set_size=2_048,
   401:         max_steps=4_000,
   402:         log_interval=100,
   403:         min_steps_before_stop=400,
   404:         early_stop_windows=3,
   405:     )
   406: 
   407: 
   408: def _emit_pred(
   409:     config: TaskConfig,
   410:     seed: int,
   411:     secret_index: int,
   412:     order_index: int,
   413:     test_preds: torch.Tensor,
   414: ) -> None:
   415:     """Emit the model's held-out predictions for the host-side scorer.
   416: 
   417:     Predictions are thresholded at 0.5 (the same threshold the metric uses) and
   418:     bit-packed. We do NOT have the test labels, so we cannot (and do not) compute
   419:     the metric here.
   420:     """
   421:     import numpy as np
   422: 
   423:     pred_bits = (test_preds.numpy() >= 0.5).astype(np.uint8)
   424:     payload = base64.b64encode(np.packbits(pred_bits).tobytes()).decode("ascii")
   425:     print(
   426:         "PARITY_PRED "
   427:         f"config={_config_tag(config)} seed={seed} secret={secret_index} "
   428:         f"order={order_index} n={int(test_preds.numel())} preds={payload}",
   429:         flush=True,
   430:     )
   431: 
   432: 
   433: def run_benchmark(
   434:     config: TaskConfig,
   435:     seed: int,
   436:     device: torch.device,
   437: ) -> dict[str, object]:
   438:     print(
   439:         "TASK_CONFIG "
   440:         + " ".join(
   441:             [
   442:                 f"N={config.n_features}",
   443:                 f"K={config.secret_size}",
   444:                 f"W={config.hidden_width}",
   445:                 f"num_hidden_secrets={config.num_hidden_secrets}",
   446:                 f"num_orderings={config.num_orderings}",
   447:                 f"test_set_size={config.test_set_size}",
   448:                 f"batch_size={config.batch_size}",
   449:                 f"max_steps={config.max_steps}",
   450:             ]
   451:         ),
   452:         flush=True,
   453:     )
   454: 
   455:     results: list[RunResult] = []
   456: 
   457:     for secret_index in range(config.num_hidden_secrets):
   458:         train_dataset_seed = seed * 10_000 + secret_index
   459:         x_pool = gen_train_pool_x(config, train_dataset_seed)
   460:         y_pool = load_train_labels(config, seed, secret_index)
   461:         train_x, train_y = normalize_dataset(
   462:             make_dataset(x_pool, y_pool, config),
   463:             config,
   464:         )
   465:         test_x = gen_test_x(config, seed * 20_000 + secret_index)
   466:         positive_rate = float(train_y.mean().item())
   467:         print(
   468:             "DATASET_METRICS "
   469:             f"secret={secret_index} num_examples={train_x.shape[0]} "
   470:             f"positive_rate={positive_rate:.6f}",
   471:             flush=True,
   472:         )
   473: 
   474:         for order_index in range(config.num_orderings):
   475:             run_seed = seed * 1_000_000 + secret_index * 1_000 + order_index
   476:             order_seed = seed * 2_000_000 + secret_index * 1_000 + order_index
   477:             result, test_preds = train_one_run(
   478:                 train_x=train_x,
   479:                 train_y=train_y,
   480:                 test_x=test_x,
   481:                 config=config,
   482:                 device=device,
   483:                 run_seed=run_seed,
   484:                 order_seed=order_seed,
   485:                 secret_index=secret_index,
   486:                 order_index=order_index,
   487:             )
   488:             results.append(result)
   489:             _emit_pred(config, seed, secret_index, order_index, test_preds)
   490: 
   491:     step_tensor = torch.tensor([result.steps for result in results], dtype=torch.float64)
   492:     print(
   493:         "BENCH_DONE "
   494:         f"num_runs={len(results)} mean_steps={float(step_tensor.mean().item()):.6f}",
   495:         flush=True,
   496:     )
   497:     return {
   498:         "config": asdict(config),
   499:         "results": [asdict(result) for result in results],
   500:     }
   501: 
   502: 
   503: def parse_args() -> argparse.Namespace:
   504:     parser = argparse.ArgumentParser(description="Run the MLS-Bench optimization-parity task.")
   505:     parser.add_argument("--seed", type=int, default=42, help="Top-level benchmark seed.")
   506:     parser.add_argument(
   507:         "--output-dir",
   508:         type=Path,
   509:         default=None,
   510:         help="Optional directory for a JSON summary.",
   511:     )
   512:     parser.add_argument(
   513:         "--label",
   514:         type=str,
   515:         default="eval",
   516:         help="Optional label stored in the JSON summary.",
   517:     )
   518:     parser.add_argument(
   519:         "--device",
   520:         choices=("auto", "cpu", "cuda"),
   521:         default="auto",
   522:         help="Execution device.",
   523:     )
   524:     parser.add_argument(
   525:         "--smoke",
   526:         action="store_true",
   527:         help="Run a smaller local sanity check without changing the benchmark defaults in code.",
   528:     )
   529:     parser.add_argument(
   530:         "--n-features",
   531:         type=int,
   532:         default=None,
   533:         help="Override n_features in TaskConfig.",
   534:     )
   535:     parser.add_argument(
   536:         "--secret-size",
   537:         type=int,
   538:         default=None,
   539:         help="Override secret_size in TaskConfig.",
   540:     )
   541:     return parser.parse_args()
   542: 
   543: 
   544: def main() -> None:
   545:     args = parse_args()
   546:     config = maybe_apply_smoke_mode(DEFAULT_TASK, args.smoke)
   547:     if args.n_features is not None:
   548:         config = replace(config, n_features=args.n_features)
   549:     if args.secret_size is not None:
   550:         config = replace(config, secret_size=args.secret_size)
   551:     device = resolve_device(args.device)
   552:     summary = run_benchmark(config=config, seed=args.seed, device=device)
   553: 
   554:     if args.output_dir is not None:
   555:         args.output_dir.mkdir(parents=True, exist_ok=True)
   556:         output_path = args.output_dir / f"{args.label}_seed{args.seed}.json"
   557:         output_path.write_text(json.dumps(summary, indent=2))
   558: 
   559: 
   560: if __name__ == "__main__":
   561:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `default` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-examples/optimization_parity/custom_strategy.py`:

```python
Lines 242–268:
   239: # =====================================================================
   240: # EDITABLE: init_model, make_dataset, get_optimizer_config
   241: # =====================================================================
   242: def init_model(model: nn.Sequential, config: TaskConfig) -> None:
   243:     """Initialize the fixed two-layer MLP."""
   244:     for layer in model:
   245:         if isinstance(layer, nn.Linear):
   246:             gain = nn.init.calculate_gain("relu") if layer is model[0] else 1.0
   247:             nn.init.xavier_uniform_(layer.weight, gain=gain)
   248:             nn.init.zeros_(layer.bias)
   249: 
   250: 
   251: def make_dataset(
   252:     x_pool: torch.Tensor,
   253:     y_pool: torch.Tensor,
   254:     config: TaskConfig,
   255: ) -> tuple[torch.Tensor, torch.Tensor]:
   256:     """Return the maximal slice of the labeled pool to induce one-pass training."""
   257:     num_examples = config.max_train_examples
   258:     return x_pool[:num_examples], y_pool[:num_examples]
   259: 
   260: 
   261: def get_optimizer_config(config: TaskConfig) -> dict[str, float]:
   262:     """Return AdamW hyperparameters for the fixed training loop."""
   263:     return {
   264:         "lr": 1e-3,
   265:         "wd": 1e-2,
   266:         "beta1": 0.9,
   267:         "beta2": 0.999,
   268:     }
   269: 
   270: 
   271: # =====================================================================
```

### `multi_epoch` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-examples/optimization_parity/custom_strategy.py`:

```python
Lines 242–269:
   239: # =====================================================================
   240: # EDITABLE: init_model, make_dataset, get_optimizer_config
   241: # =====================================================================
   242: def init_model(model: nn.Sequential, config: TaskConfig) -> None:
   243:     """Initialize the fixed two-layer MLP."""
   244:     for layer in model:
   245:         if isinstance(layer, nn.Linear):
   246:             gain = nn.init.calculate_gain("relu") if layer is model[0] else 1.0
   247:             nn.init.xavier_uniform_(layer.weight, gain=gain)
   248:             nn.init.zeros_(layer.bias)
   249: 
   250: 
   251: def make_dataset(
   252:     x_pool: torch.Tensor,
   253:     y_pool: torch.Tensor,
   254:     config: TaskConfig,
   255: ) -> tuple[torch.Tensor, torch.Tensor]:
   256:     """Use a smaller, configurable slice of the labeled pool for multi-epoch reuse."""
   257:     train_examples = 10_000  # Tunable parameter for this multi-epoch baseline.
   258:     num_examples = min(train_examples, config.max_train_examples)
   259:     return x_pool[:num_examples], y_pool[:num_examples]
   260: 
   261: 
   262: def get_optimizer_config(config: TaskConfig) -> dict[str, float]:
   263:     """Return AdamW hyperparameters for the fixed training loop."""
   264:     return {
   265:         "lr": 1e-3,
   266:         "wd": 1e-2,
   267:         "beta1": 0.9,
   268:         "beta2": 0.999,
   269:     }
   270: 
   271: 
   272: # =====================================================================
```

### `nowd` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-examples/optimization_parity/custom_strategy.py`:

```python
Lines 242–268:
   239: # =====================================================================
   240: # EDITABLE: init_model, make_dataset, get_optimizer_config
   241: # =====================================================================
   242: def init_model(model: nn.Sequential, config: TaskConfig) -> None:
   243:     """Initialize the fixed two-layer MLP."""
   244:     for layer in model:
   245:         if isinstance(layer, nn.Linear):
   246:             gain = nn.init.calculate_gain("relu") if layer is model[0] else 1.0
   247:             nn.init.xavier_uniform_(layer.weight, gain=gain)
   248:             nn.init.zeros_(layer.bias)
   249: 
   250: 
   251: def make_dataset(
   252:     x_pool: torch.Tensor,
   253:     y_pool: torch.Tensor,
   254:     config: TaskConfig,
   255: ) -> tuple[torch.Tensor, torch.Tensor]:
   256:     """Return the maximal slice of the labeled pool to induce one-pass training."""
   257:     num_examples = config.max_train_examples
   258:     return x_pool[:num_examples], y_pool[:num_examples]
   259: 
   260: 
   261: def get_optimizer_config(config: TaskConfig) -> dict[str, float]:
   262:     """Return AdamW hyperparameters with no weight decay."""
   263:     return {
   264:         "lr": 1e-3,
   265:         "wd": 0.0,
   266:         "beta1": 0.9,
   267:         "beta2": 0.999,
   268:     }
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
