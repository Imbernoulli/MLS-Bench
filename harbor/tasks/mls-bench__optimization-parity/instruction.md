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
2. `make_dataset(x_pool, config)`
3. `get_optimizer_config(config)`

## Fixed Setup
- Task: `y = (sum_{i in S} x_i) mod 2` for a hidden secret subset `S`.
- Inputs: binary vectors `x in {0, 1}^N`.
- The model architecture, optimizer family (`AdamW`), loss, batch size, training loop, and evaluation protocol are fixed by the harness and not editable. All sizes are provided to your functions via the `config` (`TaskConfig`) object.

## Interface Notes
- `init_model(...)` must not depend on the hidden secret.
- `make_dataset(x_pool, config)` sees only the **unlabeled** pool `x_pool` and returns a 1-D integer tensor of row indices into it (repeats allowed for reweighting; any order). The harness attaches the held-out labels to exactly the rows you pick — your code never sees a label or the hidden secret `S`, so it cannot recover `S` from labels; making gradient training learn the parity is the problem.
- Selected indices must lie in `[0, pool_size)`; the number of selected rows must stay `<= 12_800_000`.
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
- editable lines **306–341**




## Readable Context


### `pytorch-examples/optimization_parity/custom_strategy.py`  [EDITABLE — lines 306–341 only]

```python
     1: """Optimization-parity scaffold for MLS-Bench.
     2: 
     3: The fixed evaluation trains a fixed two-layer MLP to learn hidden sparse parity
     4: functions and asks the agent to control only:
     5:   1. model initialization
     6:   2. training-data selection (choosing which pool rows to train on)
     7:   3. AdamW hyperparameters
     8: 
     9: NOTE: The hidden parity secret S, the training-pool LABELS, and the held-out
    10: test labels are NOT visible to your editable hooks. The harness pre-generates
    11: the (unlabeled) training inputs and their labels; a FIXED driver loads the
    12: labels into its own memory (scrubbing them from disk when the harness marks
    13: them ephemeral), then hands your ``make_dataset`` only the UNLABELED pool and
    14: attaches the held-out labels to the rows you pick. Your hooks only ever see
    15: binary inputs — never a label, never S, never the test labels. The runner
    16: trains your model and emits its predictions on a held-out test set; the host
    17: regenerates the test labels and computes accuracy. A strategy must make
    18: gradient training learn the parity — it cannot recover the secret from labels.
    19: """
    20: 
    21: from __future__ import annotations
    22: 
    23: import argparse
    24: import base64
    25: import json
    26: import os
    27: import random
    28: from dataclasses import asdict, dataclass, replace
    29: from pathlib import Path
    30: 
    31: import torch
    32: from torch import nn
    33: 
    34: 
    35: # =====================================================================
    36: # FIXED: Benchmark configuration
    37: # =====================================================================
    38: @dataclass(frozen=True)
    39: class TaskConfig:
    40:     n_features: int = 32
    41:     secret_size: int = 8
    42:     hidden_width: int = 512
    43:     batch_size: int = 128
    44:     max_steps: int = 30_000
    45:     max_train_examples: int = 12_800_000
    46:     num_hidden_secrets: int = 5
    47:     num_orderings: int = 3
    48:     test_set_size: int = 16_384
    49:     log_interval: int = 250
    50:     min_steps_before_stop: int = 1_000
    51:     early_stop_acc: float = 0.999
    52:     early_stop_windows: int = 4
    53: 
    54: 
    55: @dataclass(frozen=True)
    56: class OptimizerConfig:
    57:     lr: float
    58:     wd: float
    59:     beta1: float
    60:     beta2: float
    61: 
    62: 
    63: @dataclass(frozen=True)
    64: class RunResult:
    65:     secret_index: int
    66:     order_index: int
    67:     steps: int
    68: 
    69: 
    70: DEFAULT_TASK = TaskConfig()
    71: 
    72: 
    73: def build_model(config: TaskConfig) -> nn.Sequential:
    74:     return nn.Sequential(
    75:         nn.Linear(config.n_features, config.hidden_width),
    76:         nn.ReLU(),
    77:         nn.Linear(config.hidden_width, 1),
    78:         nn.Sigmoid(),
    79:     )
    80: 
    81: 
    82: def set_global_seed(seed: int) -> None:
    83:     random.seed(seed)
    84:     torch.manual_seed(seed)
    85:     if torch.cuda.is_available():
    86:         torch.cuda.manual_seed_all(seed)
    87: 
    88: 
    89: def normalize_dataset(
    90:     dataset: object,
    91:     config: TaskConfig,
    92: ) -> tuple[torch.Tensor, torch.Tensor]:
    93:     if isinstance(dataset, dict):
    94:         if "x" not in dataset or "y" not in dataset:
    95:             raise ValueError("Dataset dict must contain 'x' and 'y'.")
    96:         x, y = dataset["x"], dataset["y"]
    97:     elif isinstance(dataset, (tuple, list)) and len(dataset) == 2:
    98:         x, y = dataset
    99:     else:
   100:         raise TypeError("Dataset must be a (x, y) pair or a dict with keys 'x' and 'y'.")
   101: 
   102:     x = torch.as_tensor(x, dtype=torch.float32)
   103:     y = torch.as_tensor(y, dtype=torch.float32).view(-1)
   104: 
   105:     if x.ndim != 2:
   106:         raise ValueError(f"Expected x to have shape [num_examples, n_features], got {tuple(x.shape)}.")
   107:     if x.shape[1] != config.n_features:
   108:         raise ValueError(
   109:             f"Expected x.shape[1] == {config.n_features}, got {x.shape[1]}."
   110:         )
   111:     if x.shape[0] != y.shape[0]:
   112:         raise ValueError("x and y must contain the same number of examples.")
   113:     if x.shape[0] == 0:
   114:         raise ValueError("Training dataset must contain at least one example.")
   115:     if x.shape[0] > config.max_train_examples:
   116:         raise ValueError(
   117:             f"Training dataset size {x.shape[0]} exceeds limit {config.max_train_examples}."
   118:         )
   119:     if not torch.all((x == 0) | (x == 1)):
   120:         raise ValueError("Training inputs must stay in {0, 1}.")
   121:     if not torch.all((y == 0) | (y == 1)):
   122:         raise ValueError("Training labels must stay in {0, 1}.")
   123:     return x.contiguous(), y.contiguous()
   124: 
   125: 
   126: def normalize_optimizer_config(config_dict: dict[str, float]) -> OptimizerConfig:
   127:     required = {"lr", "wd", "beta1", "beta2"}
   128:     missing = required - set(config_dict)
   129:     if missing:
   130:         raise ValueError(f"Missing optimizer hyperparameters: {sorted(missing)}")
   131: 
   132:     config = OptimizerConfig(
   133:         lr=float(config_dict["lr"]),
   134:         wd=float(config_dict["wd"]),
   135:         beta1=float(config_dict["beta1"]),
   136:         beta2=float(config_dict["beta2"]),
   137:     )
   138:     if not config.lr > 0.0:
   139:         raise ValueError("AdamW learning rate must be positive.")
   140:     if not config.wd >= 0.0:
   141:         raise ValueError("AdamW weight decay must be non-negative.")
   142:     if not 0.0 < config.beta1 < 1.0:
   143:         raise ValueError("AdamW beta1 must satisfy 0 < beta1 < 1.")
   144:     if not 0.0 < config.beta2 < 1.0:
   145:         raise ValueError("AdamW beta2 must satisfy 0 < beta2 < 1.")
   146:     return config
   147: 
   148: 
   149: def predict_on(
   150:     model: nn.Module,
   151:     x: torch.Tensor,
   152:     device: torch.device,
   153:     batch_size: int = 4096,
   154: ) -> torch.Tensor:
   155:     """Return raw model outputs (sigmoid probabilities) for every row of x."""
   156:     model.eval()
   157:     outputs = []
   158:     with torch.no_grad():
   159:         for start in range(0, x.shape[0], batch_size):
   160:             end = start + batch_size
   161:             batch_x = x[start:end].to(device)
   162:             preds = model(batch_x).view(-1)
   163:             outputs.append(preds.detach().cpu())
   164:     return torch.cat(outputs) if outputs else torch.empty(0)
   165: 
   166: 
   167: def maybe_log_final_window(
   168:     secret_index: int,
   169:     order_index: int,
   170:     steps: int,
   171:     window_loss: float,
   172:     window_acc: float,
   173:     window_count: int,
   174: ) -> None:
   175:     if window_count == 0:
   176:         return
   177:     print(
   178:         "TRAIN_METRICS "
   179:         f"secret={secret_index} order={order_index} step={steps} "
   180:         f"loss={window_loss / window_count:.6f} acc={window_acc / window_count:.6f}",
   181:         flush=True,
   182:     )
   183: 
   184: 
   185: # =====================================================================
   186: # FIXED: held-out input loading (the harness pre-generates these; the
   187: # secret and the test labels are never present in this process). The
   188: # training-pool LABELS are loaded here, in fixed code — and scrubbed from
   189: # disk when the harness marks them ephemeral (MLSBENCH_EPHEMERAL_INPUTS=1)
   190: # — so make_dataset() below only ever sees the UNLABELED pool.
   191: # =====================================================================
   192: def _inputs_dir() -> str:
   193:     """Directory holding the pre-generated parity inputs for this task."""
   194:     env = os.environ.get("PARITY_INPUTS_DIR")
   195:     if env:
   196:         return env
   197:     return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_parity_inputs")
   198: 
   199: 
   200: def _config_tag(config: TaskConfig) -> str:
   201:     return f"n{config.n_features}_k{config.secret_size}"
   202: 
   203: 
   204: def gen_train_pool_x(config: TaskConfig, train_dataset_seed: int) -> torch.Tensor:
   205:     """Regenerate the full unlabeled training x-pool (no secret involved)."""
   206:     generator = torch.Generator().manual_seed(train_dataset_seed)
   207:     return torch.randint(
   208:         low=0,
   209:         high=2,
   210:         size=(config.max_train_examples, config.n_features),
   211:         generator=generator,
   212:         dtype=torch.int64,
   213:     ).to(torch.float32)
   214: 
   215: 
   216: def gen_test_x(config: TaskConfig, test_seed: int) -> torch.Tensor:
   217:     """Regenerate the held-out test inputs (no secret; labels are withheld)."""
   218:     generator = torch.Generator().manual_seed(test_seed)
   219:     return torch.randint(
   220:         low=0,
   221:         high=2,
   222:         size=(config.test_set_size, config.n_features),
   223:         generator=generator,
   224:         dtype=torch.int64,
   225:     ).to(torch.float32)
   226: 
   227: 
   228: def load_train_labels(config: TaskConfig, seed: int, secret_index: int) -> torch.Tensor:
   229:     """Load the bit-packed training-pool labels for one hidden secret.
   230: 
   231:     Only the labels are provided (the secret that produced them is held out).
   232:     The labels are bit-packed for one row per training example over the full
   233:     ``max_train_examples`` pool; unpack to a float tensor in {0, 1}.
   234: 
   235:     This is FIXED code, called only by ``_load_all_train_labels`` below, which
   236:     deletes the on-disk blob afterward when the harness marks the inputs as
   237:     ephemeral. It is never invoked from an editable hook.
   238:     """
   239:     import numpy as np
   240: 
   241:     tag = _config_tag(config)
   242:     path = os.path.join(_inputs_dir(), f"{tag}_seed{seed}_s{secret_index}.labels.b64")
   243:     with open(path, "r") as f:
   244:         packed = np.frombuffer(base64.b64decode(f.read()), dtype=np.uint8)
   245:     bits = np.unpackbits(packed)[: config.max_train_examples]
   246:     return torch.from_numpy(bits.astype("float32"))
   247: 
   248: 
   249: def _load_all_train_labels(config: TaskConfig, seed: int) -> dict[int, torch.Tensor]:
   250:     """Load every hidden secret's training-pool labels into memory, then DELETE
   251:     the on-disk label blobs when the harness marks the materialized inputs as
   252:     ephemeral (MLSBENCH_EPHEMERAL_INPUTS=1, i.e. re-created before every
   253:     evaluation). The delete keeps parity honest there: the editable
   254:     ``make_dataset`` hook (which runs later) cannot reopen the blobs to recover
   255:     the hidden secret, so the strategy must help gradient training learn the
   256:     parity. Natively (no ENV set) the blobs persist — they are staged once per
   257:     workspace and must survive across evaluations."""
   258:     labels: dict[int, torch.Tensor] = {}
   259:     for secret_index in range(config.num_hidden_secrets):
   260:         labels[secret_index] = load_train_labels(config, seed, secret_index)
   261:     if os.environ.get("MLSBENCH_EPHEMERAL_INPUTS") == "1":
   262:         tag = _config_tag(config)
   263:         inputs_dir = _inputs_dir()
   264:         for secret_index in range(config.num_hidden_secrets):
   265:             blob = os.path.join(inputs_dir, f"{tag}_seed{seed}_s{secret_index}.labels.b64")
   266:             try:
   267:                 os.remove(blob)
   268:             except OSError:
   269:                 pass
   270:     return labels
   271: 
   272: 
   273: def _resolve_indices(
   274:     selection: object,
   275:     pool_size: int,
   276:     config: TaskConfig,
   277: ) -> torch.Tensor:
   278:     """Validate the editable ``make_dataset`` output.
   279: 
   280:     It must be a 1-D collection of integer row indices into the unlabeled pool
   281:     (repeats allowed for reweighting; any order). Fixed code then attaches the
   282:     held-out labels to exactly those rows — the strategy never picks a label.
   283:     """
   284:     idx = torch.as_tensor(selection).reshape(-1)
   285:     if not torch.is_floating_point(idx):
   286:         idx = idx.long()
   287:     else:
   288:         if not torch.all(idx == idx.long().to(idx.dtype)):
   289:             raise ValueError("make_dataset indices must be integers.")
   290:         idx = idx.long()
   291:     if idx.numel() == 0:
   292:         raise ValueError("make_dataset must return at least one row index.")
   293:     if idx.numel() > config.max_train_examples:
   294:         raise ValueError(
   295:             f"make_dataset selected {idx.numel()} rows, exceeds limit "
   296:             f"{config.max_train_examples}."
   297:         )
   298:     if int(idx.min()) < 0 or int(idx.max()) >= pool_size:
   299:         raise ValueError("make_dataset indices out of range for the training pool.")
   300:     return idx.contiguous()
   301: 
   302: 
   303: # =====================================================================
   304: # EDITABLE: init_model, make_dataset, get_optimizer_config
   305: # =====================================================================
   306: def init_model(model: nn.Sequential, config: TaskConfig) -> None:
   307:     """Initialize the fixed two-layer MLP."""
   308:     for layer in model:
   309:         if isinstance(layer, nn.Linear):
   310:             gain = nn.init.calculate_gain("relu") if layer is model[0] else 1.0
   311:             nn.init.xavier_uniform_(layer.weight, gain=gain)
   312:             nn.init.zeros_(layer.bias)
   313: 
   314: 
   315: def make_dataset(
   316:     x_pool: torch.Tensor,
   317:     config: TaskConfig,
   318: ) -> torch.Tensor:
   319:     """Select which training rows to use — you see only the UNLABELED pool.
   320: 
   321:     ``x_pool`` is a large pool of binary inputs (shape [pool, n_features]) drawn
   322:     from the same distribution as the held-out test set. Return a 1-D LongTensor
   323:     of row indices into ``x_pool`` (repeats allowed for reweighting; any order).
   324:     The harness attaches the held-out labels to exactly the rows you pick and
   325:     trains on them.
   326: 
   327:     You never see the labels or the hidden parity secret, so a strategy must make
   328:     gradient training learn the parity — it cannot solve the secret from labels.
   329:     """
   330:     num_examples = 4_096
   331:     return torch.arange(num_examples)
   332: 
   333: 
   334: def get_optimizer_config(config: TaskConfig) -> dict[str, float]:
   335:     """Return AdamW hyperparameters for the fixed training loop."""
   336:     return {
   337:         "lr": 1e-3,
   338:         "wd": 1e-2,
   339:         "beta1": 0.9,
   340:         "beta2": 0.999,
   341:     }
   342: 
   343: 
   344: # =====================================================================
   345: # FIXED: training and prediction driver
   346: # =====================================================================
   347: def train_one_run(
   348:     train_x: torch.Tensor,
   349:     train_y: torch.Tensor,
   350:     test_x: torch.Tensor,
   351:     config: TaskConfig,
   352:     device: torch.device,
   353:     run_seed: int,
   354:     order_seed: int,
   355:     secret_index: int,
   356:     order_index: int,
   357: ) -> tuple[RunResult, torch.Tensor]:
   358:     set_global_seed(run_seed)
   359: 
   360:     model = build_model(config).to(device)
   361:     init_model(model, config)
   362:     optimizer_config = normalize_optimizer_config(get_optimizer_config(config))
   363:     optimizer = torch.optim.AdamW(
   364:         model.parameters(),
   365:         lr=optimizer_config.lr,
   366:         betas=(optimizer_config.beta1, optimizer_config.beta2),
   367:         weight_decay=optimizer_config.wd,
   368:     )
   369:     criterion = nn.BCELoss()
   370: 
   371:     steps = 0
   372:     stable_windows = 0
   373:     window_loss = 0.0
   374:     window_acc = 0.0
   375:     window_count = 0
   376:     last_logged_step = 0
   377:     permutation_generator = torch.Generator().manual_seed(order_seed)
   378: 
   379:     while steps < config.max_steps:
   380:         permutation = torch.randperm(train_x.shape[0], generator=permutation_generator)
   381:         for start in range(0, train_x.shape[0], config.batch_size):
   382:             batch_indices = permutation[start : start + config.batch_size]
   383:             batch_x = train_x.index_select(0, batch_indices).to(device)
   384:             batch_y = train_y.index_select(0, batch_indices).to(device)
   385: 
   386:             optimizer.zero_grad(set_to_none=True)
   387:             preds = model(batch_x).view(-1)
   388:             loss = criterion(preds, batch_y)
   389:             loss.backward()
   390:             optimizer.step()
   391: 
   392:             batch_acc = ((preds >= 0.5) == (batch_y >= 0.5)).float().mean().item()
   393:             window_loss += loss.item()
   394:             window_acc += batch_acc
   395:             window_count += 1
   396:             steps += 1
   397: 
   398:             should_log = steps == 1 or steps % config.log_interval == 0 or steps == config.max_steps
   399:             if should_log:
   400:                 avg_loss = window_loss / window_count
   401:                 avg_acc = window_acc / window_count
   402:                 print(
   403:                     "TRAIN_METRICS "
   404:                     f"secret={secret_index} order={order_index} step={steps} "
   405:                     f"loss={avg_loss:.6f} acc={avg_acc:.6f}",
   406:                     flush=True,
   407:                 )
   408:                 last_logged_step = steps
   409:                 if steps >= config.min_steps_before_stop and avg_acc >= config.early_stop_acc:
   410:                     stable_windows += 1
   411:                 else:
   412:                     stable_windows = 0
   413:                 window_loss = 0.0
   414:                 window_acc = 0.0
   415:                 window_count = 0
   416:                 if stable_windows >= config.early_stop_windows:
   417:                     break
   418: 
   419:             if steps >= config.max_steps:
   420:                 break
   421:         if stable_windows >= config.early_stop_windows or steps >= config.max_steps:
   422:             break
   423: 
   424:     if last_logged_step != steps:
   425:         maybe_log_final_window(
   426:             secret_index=secret_index,
   427:             order_index=order_index,
   428:             steps=steps,
   429:             window_loss=window_loss,
   430:             window_acc=window_acc,
   431:             window_count=window_count,
   432:         )
   433: 
   434:     test_preds = predict_on(model, test_x, device)
   435:     print(
   436:         "RUN_METRICS "
   437:         f"secret={secret_index} order={order_index} steps={steps}",
   438:         flush=True,
   439:     )
   440:     return (
   441:         RunResult(
   442:             secret_index=secret_index,
   443:             order_index=order_index,
   444:             steps=steps,
   445:         ),
   446:         test_preds,
   447:     )
   448: 
   449: 
   450: def resolve_device(device_arg: str) -> torch.device:
   451:     if device_arg == "cpu":
   452:         return torch.device("cpu")
   453:     if device_arg == "cuda":
   454:         if not torch.cuda.is_available():
   455:             raise RuntimeError("CUDA requested but no GPU is available.")
   456:         return torch.device("cuda")
   457:     return torch.device("cuda" if torch.cuda.is_available() else "cpu")
   458: 
   459: 
   460: def maybe_apply_smoke_mode(config: TaskConfig, enabled: bool) -> TaskConfig:
   461:     if not enabled:
   462:         return config
   463:     return replace(
   464:         config,
   465:         num_hidden_secrets=2,
   466:         num_orderings=2,
   467:         test_set_size=2_048,
   468:         max_steps=4_000,
   469:         log_interval=100,
   470:         min_steps_before_stop=400,
   471:         early_stop_windows=3,
   472:     )
   473: 
   474: 
   475: def _emit_pred(
   476:     config: TaskConfig,
   477:     seed: int,
   478:     secret_index: int,
   479:     order_index: int,
   480:     test_preds: torch.Tensor,
   481: ) -> None:
   482:     """Emit the model's held-out predictions for the host-side scorer.
   483: 
   484:     Predictions are thresholded at 0.5 (the same threshold the metric uses) and
   485:     bit-packed. We do NOT have the test labels, so we cannot (and do not) compute
   486:     the metric here.
   487:     """
   488:     import numpy as np
   489: 
   490:     pred_bits = (test_preds.numpy() >= 0.5).astype(np.uint8)
   491:     payload = base64.b64encode(np.packbits(pred_bits).tobytes()).decode("ascii")
   492:     print(
   493:         "PARITY_PRED "
   494:         f"config={_config_tag(config)} seed={seed} secret={secret_index} "
   495:         f"order={order_index} n={int(test_preds.numel())} preds={payload}",
   496:         flush=True,
   497:     )
   498: 
   499: 
   500: def run_benchmark(
   501:     config: TaskConfig,
   502:     seed: int,
   503:     device: torch.device,
   504: ) -> dict[str, object]:
   505:     print(
   506:         "TASK_CONFIG "
   507:         + " ".join(
   508:             [
   509:                 f"N={config.n_features}",
   510:                 f"K={config.secret_size}",
   511:                 f"W={config.hidden_width}",
   512:                 f"num_hidden_secrets={config.num_hidden_secrets}",
   513:                 f"num_orderings={config.num_orderings}",
   514:                 f"test_set_size={config.test_set_size}",
   515:                 f"batch_size={config.batch_size}",
   516:                 f"max_steps={config.max_steps}",
   517:             ]
   518:         ),
   519:         flush=True,
   520:     )
   521: 
   522:     # FIXED: load every secret's pool labels into memory (scrubbing the blobs
   523:     # when the harness marks them ephemeral) before any editable hook runs.
   524:     labels_by_secret = _load_all_train_labels(config, seed)
   525: 
   526:     results: list[RunResult] = []
   527: 
   528:     for secret_index in range(config.num_hidden_secrets):
   529:         train_dataset_seed = seed * 10_000 + secret_index
   530:         x_pool = gen_train_pool_x(config, train_dataset_seed)
   531:         y_pool = labels_by_secret[secret_index]
   532:         # The editable hook only sees the UNLABELED pool and returns row indices;
   533:         # fixed code attaches the held-out labels to exactly those rows.
   534:         selected = _resolve_indices(make_dataset(x_pool, config), x_pool.shape[0], config)
   535:         train_x, train_y = normalize_dataset(
   536:             (x_pool.index_select(0, selected), y_pool.index_select(0, selected)),
   537:             config,
   538:         )
   539:         test_x = gen_test_x(config, seed * 20_000 + secret_index)
   540:         positive_rate = float(train_y.mean().item())
   541:         print(
   542:             "DATASET_METRICS "
   543:             f"secret={secret_index} num_examples={train_x.shape[0]} "
   544:             f"positive_rate={positive_rate:.6f}",
   545:             flush=True,
   546:         )
   547: 
   548:         for order_index in range(config.num_orderings):
   549:             run_seed = seed * 1_000_000 + secret_index * 1_000 + order_index
   550:             order_seed = seed * 2_000_000 + secret_index * 1_000 + order_index
   551:             result, test_preds = train_one_run(
   552:                 train_x=train_x,
   553:                 train_y=train_y,
   554:                 test_x=test_x,
   555:                 config=config,
   556:                 device=device,
   557:                 run_seed=run_seed,
   558:                 order_seed=order_seed,
   559:                 secret_index=secret_index,
   560:                 order_index=order_index,
   561:             )
   562:             results.append(result)
   563:             _emit_pred(config, seed, secret_index, order_index, test_preds)
   564: 
   565:     step_tensor = torch.tensor([result.steps for result in results], dtype=torch.float64)
   566:     print(
   567:         "BENCH_DONE "
   568:         f"num_runs={len(results)} mean_steps={float(step_tensor.mean().item()):.6f}",
   569:         flush=True,
   570:     )
   571:     return {
   572:         "config": asdict(config),
   573:         "results": [asdict(result) for result in results],
   574:     }
   575: 
   576: 
   577: def parse_args() -> argparse.Namespace:
   578:     parser = argparse.ArgumentParser(description="Run the MLS-Bench optimization-parity task.")
   579:     parser.add_argument("--seed", type=int, default=42, help="Top-level benchmark seed.")
   580:     parser.add_argument(
   581:         "--output-dir",
   582:         type=Path,
   583:         default=None,
   584:         help="Optional directory for a JSON summary.",
   585:     )
   586:     parser.add_argument(
   587:         "--label",
   588:         type=str,
   589:         default="eval",
   590:         help="Optional label stored in the JSON summary.",
   591:     )
   592:     parser.add_argument(
   593:         "--device",
   594:         choices=("auto", "cpu", "cuda"),
   595:         default="auto",
   596:         help="Execution device.",
   597:     )
   598:     parser.add_argument(
   599:         "--smoke",
   600:         action="store_true",
   601:         help="Run a smaller local sanity check without changing the benchmark defaults in code.",
   602:     )
   603:     parser.add_argument(
   604:         "--n-features",
   605:         type=int,
   606:         default=None,
   607:         help="Override n_features in TaskConfig.",
   608:     )
   609:     parser.add_argument(
   610:         "--secret-size",
   611:         type=int,
   612:         default=None,
   613:         help="Override secret_size in TaskConfig.",
   614:     )
   615:     return parser.parse_args()
   616: 
   617: 
   618: def main() -> None:
   619:     args = parse_args()
   620:     config = maybe_apply_smoke_mode(DEFAULT_TASK, args.smoke)
   621:     if args.n_features is not None:
   622:         config = replace(config, n_features=args.n_features)
   623:     if args.secret_size is not None:
   624:         config = replace(config, secret_size=args.secret_size)
   625:     device = resolve_device(args.device)
   626:     summary = run_benchmark(config=config, seed=args.seed, device=device)
   627: 
   628:     if args.output_dir is not None:
   629:         args.output_dir.mkdir(parents=True, exist_ok=True)
   630:         output_path = args.output_dir / f"{args.label}_seed{args.seed}.json"
   631:         output_path.write_text(json.dumps(summary, indent=2))
   632: 
   633: 
   634: if __name__ == "__main__":
   635:     main()
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
Lines 306–331:
   303: # =====================================================================
   304: # EDITABLE: init_model, make_dataset, get_optimizer_config
   305: # =====================================================================
   306: def init_model(model: nn.Sequential, config: TaskConfig) -> None:
   307:     """Initialize the fixed two-layer MLP."""
   308:     for layer in model:
   309:         if isinstance(layer, nn.Linear):
   310:             gain = nn.init.calculate_gain("relu") if layer is model[0] else 1.0
   311:             nn.init.xavier_uniform_(layer.weight, gain=gain)
   312:             nn.init.zeros_(layer.bias)
   313: 
   314: 
   315: def make_dataset(
   316:     x_pool: torch.Tensor,
   317:     config: TaskConfig,
   318: ) -> torch.Tensor:
   319:     """Return the maximal prefix of the (unlabeled) pool to induce one-pass training."""
   320:     num_examples = config.max_train_examples
   321:     return torch.arange(num_examples)
   322: 
   323: 
   324: def get_optimizer_config(config: TaskConfig) -> dict[str, float]:
   325:     """Return AdamW hyperparameters for the fixed training loop."""
   326:     return {
   327:         "lr": 1e-3,
   328:         "wd": 1e-2,
   329:         "beta1": 0.9,
   330:         "beta2": 0.999,
   331:     }
   332: 
   333: 
   334: # =====================================================================
```

### `multi_epoch` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-examples/optimization_parity/custom_strategy.py`:

```python
Lines 306–332:
   303: # =====================================================================
   304: # EDITABLE: init_model, make_dataset, get_optimizer_config
   305: # =====================================================================
   306: def init_model(model: nn.Sequential, config: TaskConfig) -> None:
   307:     """Initialize the fixed two-layer MLP."""
   308:     for layer in model:
   309:         if isinstance(layer, nn.Linear):
   310:             gain = nn.init.calculate_gain("relu") if layer is model[0] else 1.0
   311:             nn.init.xavier_uniform_(layer.weight, gain=gain)
   312:             nn.init.zeros_(layer.bias)
   313: 
   314: 
   315: def make_dataset(
   316:     x_pool: torch.Tensor,
   317:     config: TaskConfig,
   318: ) -> torch.Tensor:
   319:     """Use a smaller, configurable prefix of the (unlabeled) pool for multi-epoch reuse."""
   320:     train_examples = 10_000  # Tunable parameter for this multi-epoch baseline.
   321:     num_examples = min(train_examples, config.max_train_examples)
   322:     return torch.arange(num_examples)
   323: 
   324: 
   325: def get_optimizer_config(config: TaskConfig) -> dict[str, float]:
   326:     """Return AdamW hyperparameters for the fixed training loop."""
   327:     return {
   328:         "lr": 1e-3,
   329:         "wd": 1e-2,
   330:         "beta1": 0.9,
   331:         "beta2": 0.999,
   332:     }
   333: 
   334: 
   335: # =====================================================================
```

### `nowd` baseline — editable region  [READ-ONLY — reference implementation]

In `pytorch-examples/optimization_parity/custom_strategy.py`:

```python
Lines 306–331:
   303: # =====================================================================
   304: # EDITABLE: init_model, make_dataset, get_optimizer_config
   305: # =====================================================================
   306: def init_model(model: nn.Sequential, config: TaskConfig) -> None:
   307:     """Initialize the fixed two-layer MLP."""
   308:     for layer in model:
   309:         if isinstance(layer, nn.Linear):
   310:             gain = nn.init.calculate_gain("relu") if layer is model[0] else 1.0
   311:             nn.init.xavier_uniform_(layer.weight, gain=gain)
   312:             nn.init.zeros_(layer.bias)
   313: 
   314: 
   315: def make_dataset(
   316:     x_pool: torch.Tensor,
   317:     config: TaskConfig,
   318: ) -> torch.Tensor:
   319:     """Return the maximal prefix of the (unlabeled) pool to induce one-pass training."""
   320:     num_examples = config.max_train_examples
   321:     return torch.arange(num_examples)
   322: 
   323: 
   324: def get_optimizer_config(config: TaskConfig) -> dict[str, float]:
   325:     """Return AdamW hyperparameters with no weight decay."""
   326:     return {
   327:         "lr": 1e-3,
   328:         "wd": 0.0,
   329:         "beta1": 0.9,
   330:         "beta2": 0.999,
   331:     }
   332: 
   333: 
   334: # =====================================================================
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
