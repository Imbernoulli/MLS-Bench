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
   229:     """Load one hidden secret's bit-packed training-pool labels (only the
   230:     labels — the secret that produced them is held out).
   231: 
   232:     Prefers the payload preloaded by the FIXED wrapper (scripts/fixed_entry.py
   233:     reads and unlinks the blobs BEFORE this module is imported); falls back to
   234:     the on-disk blob when launched directly. FIXED code, called only by
   235:     ``_load_all_train_labels`` below — never from an editable hook.
   236:     """
   237:     import numpy as np
   238: 
   239:     name = f"{_config_tag(config)}_seed{seed}_s{secret_index}.labels.b64"
   240:     payload = (_PRELOADED_INPUTS or {}).pop(name, None)
   241:     if payload is None:
   242:         with open(os.path.join(_inputs_dir(), name), "r") as f:
   243:             payload = f.read()
   244:     packed = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
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
   262:         import glob as _glob
   263:         tag = _config_tag(config)
   264:         pattern = os.path.join(_inputs_dir(), f"{tag}_seed{seed}_s*.labels.b64")
   265:         for blob in _glob.glob(pattern):
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
   347: # Set by the FIXED wrapper (scripts/fixed_entry.py) AFTER it read and
   348: # unlinked the staged label blobs and BEFORE this module was imported:
   349: # maps blob basename -> file content. None when the module is launched
   350: # directly — the loaders above then read the on-disk blobs (legacy flow).
   351: _PRELOADED_INPUTS: dict[str, str] | None = None
   352: 
   353: 
   354: def train_one_run(
   355:     train_x: torch.Tensor,
   356:     train_y: torch.Tensor,
   357:     test_x: torch.Tensor,
   358:     config: TaskConfig,
   359:     device: torch.device,
   360:     run_seed: int,
   361:     order_seed: int,
   362:     secret_index: int,
   363:     order_index: int,
   364: ) -> tuple[RunResult, torch.Tensor]:
   365:     set_global_seed(run_seed)
   366: 
   367:     model = build_model(config).to(device)
   368:     init_model(model, config)
   369:     optimizer_config = normalize_optimizer_config(get_optimizer_config(config))
   370:     optimizer = torch.optim.AdamW(
   371:         model.parameters(),
   372:         lr=optimizer_config.lr,
   373:         betas=(optimizer_config.beta1, optimizer_config.beta2),
   374:         weight_decay=optimizer_config.wd,
   375:     )
   376:     criterion = nn.BCELoss()
   377: 
   378:     steps = 0
   379:     stable_windows = 0
   380:     window_loss = 0.0
   381:     window_acc = 0.0
   382:     window_count = 0
   383:     last_logged_step = 0
   384:     permutation_generator = torch.Generator().manual_seed(order_seed)
   385: 
   386:     while steps < config.max_steps:
   387:         permutation = torch.randperm(train_x.shape[0], generator=permutation_generator)
   388:         for start in range(0, train_x.shape[0], config.batch_size):
   389:             batch_indices = permutation[start : start + config.batch_size]
   390:             batch_x = train_x.index_select(0, batch_indices).to(device)
   391:             batch_y = train_y.index_select(0, batch_indices).to(device)
   392: 
   393:             optimizer.zero_grad(set_to_none=True)
   394:             preds = model(batch_x).view(-1)
   395:             loss = criterion(preds, batch_y)
   396:             loss.backward()
   397:             optimizer.step()
   398: 
   399:             batch_acc = ((preds >= 0.5) == (batch_y >= 0.5)).float().mean().item()
   400:             window_loss += loss.item()
   401:             window_acc += batch_acc
   402:             window_count += 1
   403:             steps += 1
   404: 
   405:             should_log = steps == 1 or steps % config.log_interval == 0 or steps == config.max_steps
   406:             if should_log:
   407:                 avg_loss = window_loss / window_count
   408:                 avg_acc = window_acc / window_count
   409:                 print(
   410:                     "TRAIN_METRICS "
   411:                     f"secret={secret_index} order={order_index} step={steps} "
   412:                     f"loss={avg_loss:.6f} acc={avg_acc:.6f}",
   413:                     flush=True,
   414:                 )
   415:                 last_logged_step = steps
   416:                 if steps >= config.min_steps_before_stop and avg_acc >= config.early_stop_acc:
   417:                     stable_windows += 1
   418:                 else:
   419:                     stable_windows = 0
   420:                 window_loss = 0.0
   421:                 window_acc = 0.0
   422:                 window_count = 0
   423:                 if stable_windows >= config.early_stop_windows:
   424:                     break
   425: 
   426:             if steps >= config.max_steps:
   427:                 break
   428:         if stable_windows >= config.early_stop_windows or steps >= config.max_steps:
   429:             break
   430: 
   431:     if last_logged_step != steps:
   432:         maybe_log_final_window(
   433:             secret_index=secret_index,
   434:             order_index=order_index,
   435:             steps=steps,
   436:             window_loss=window_loss,
   437:             window_acc=window_acc,
   438:             window_count=window_count,
   439:         )
   440: 
   441:     test_preds = predict_on(model, test_x, device)
   442:     print(
   443:         "RUN_METRICS "
   444:         f"secret={secret_index} order={order_index} steps={steps}",
   445:         flush=True,
   446:     )
   447:     return (
   448:         RunResult(
   449:             secret_index=secret_index,
   450:             order_index=order_index,
   451:             steps=steps,
   452:         ),
   453:         test_preds,
   454:     )
   455: 
   456: 
   457: def resolve_device(device_arg: str) -> torch.device:
   458:     if device_arg == "cpu":
   459:         return torch.device("cpu")
   460:     if device_arg == "cuda":
   461:         if not torch.cuda.is_available():
   462:             raise RuntimeError("CUDA requested but no GPU is available.")
   463:         return torch.device("cuda")
   464:     return torch.device("cuda" if torch.cuda.is_available() else "cpu")
   465: 
   466: 
   467: def maybe_apply_smoke_mode(config: TaskConfig, enabled: bool) -> TaskConfig:
   468:     if not enabled:
   469:         return config
   470:     return replace(
   471:         config,
   472:         num_hidden_secrets=2,
   473:         num_orderings=2,
   474:         test_set_size=2_048,
   475:         max_steps=4_000,
   476:         log_interval=100,
   477:         min_steps_before_stop=400,
   478:         early_stop_windows=3,
   479:     )
   480: 
   481: 
   482: def _emit_pred(
   483:     config: TaskConfig,
   484:     seed: int,
   485:     secret_index: int,
   486:     order_index: int,
   487:     test_preds: torch.Tensor,
   488: ) -> None:
   489:     """Emit the model's held-out predictions for the host-side scorer.
   490: 
   491:     Predictions are thresholded at 0.5 (the same threshold the metric uses) and
   492:     bit-packed. We do NOT have the test labels, so we cannot (and do not) compute
   493:     the metric here.
   494:     """
   495:     import numpy as np
   496: 
   497:     pred_bits = (test_preds.numpy() >= 0.5).astype(np.uint8)
   498:     payload = base64.b64encode(np.packbits(pred_bits).tobytes()).decode("ascii")
   499:     print(
   500:         "PARITY_PRED "
   501:         f"config={_config_tag(config)} seed={seed} secret={secret_index} "
   502:         f"order={order_index} n={int(test_preds.numel())} preds={payload}",
   503:         flush=True,
   504:     )
   505: 
   506: 
   507: def run_benchmark(
   508:     config: TaskConfig,
   509:     seed: int,
   510:     device: torch.device,
   511: ) -> dict[str, object]:
   512:     print(
   513:         "TASK_CONFIG "
   514:         + " ".join(
   515:             [
   516:                 f"N={config.n_features}",
   517:                 f"K={config.secret_size}",
   518:                 f"W={config.hidden_width}",
   519:                 f"num_hidden_secrets={config.num_hidden_secrets}",
   520:                 f"num_orderings={config.num_orderings}",
   521:                 f"test_set_size={config.test_set_size}",
   522:                 f"batch_size={config.batch_size}",
   523:                 f"max_steps={config.max_steps}",
   524:             ]
   525:         ),
   526:         flush=True,
   527:     )
   528: 
   529:     # FIXED: load every secret's pool labels into memory (scrubbing the blobs
   530:     # when the harness marks them ephemeral) before any editable hook runs.
   531:     labels_by_secret = _load_all_train_labels(config, seed)
   532:     # Drop any remaining preloaded payloads: from here on the labels live
   533:     # only in fixed-driver locals, exactly as in the direct-launch flow.
   534:     global _PRELOADED_INPUTS
   535:     _PRELOADED_INPUTS = None
   536: 
   537:     results: list[RunResult] = []
   538: 
   539:     for secret_index in range(config.num_hidden_secrets):
   540:         train_dataset_seed = seed * 10_000 + secret_index
   541:         x_pool = gen_train_pool_x(config, train_dataset_seed)
   542:         y_pool = labels_by_secret[secret_index]
   543:         # The editable hook only sees the UNLABELED pool and returns row indices;
   544:         # fixed code attaches the held-out labels to exactly those rows.
   545:         selected = _resolve_indices(make_dataset(x_pool, config), x_pool.shape[0], config)
   546:         train_x, train_y = normalize_dataset(
   547:             (x_pool.index_select(0, selected), y_pool.index_select(0, selected)),
   548:             config,
   549:         )
   550:         test_x = gen_test_x(config, seed * 20_000 + secret_index)
   551:         positive_rate = float(train_y.mean().item())
   552:         print(
   553:             "DATASET_METRICS "
   554:             f"secret={secret_index} num_examples={train_x.shape[0]} "
   555:             f"positive_rate={positive_rate:.6f}",
   556:             flush=True,
   557:         )
   558: 
   559:         for order_index in range(config.num_orderings):
   560:             run_seed = seed * 1_000_000 + secret_index * 1_000 + order_index
   561:             order_seed = seed * 2_000_000 + secret_index * 1_000 + order_index
   562:             result, test_preds = train_one_run(
   563:                 train_x=train_x,
   564:                 train_y=train_y,
   565:                 test_x=test_x,
   566:                 config=config,
   567:                 device=device,
   568:                 run_seed=run_seed,
   569:                 order_seed=order_seed,
   570:                 secret_index=secret_index,
   571:                 order_index=order_index,
   572:             )
   573:             results.append(result)
   574:             _emit_pred(config, seed, secret_index, order_index, test_preds)
   575: 
   576:     step_tensor = torch.tensor([result.steps for result in results], dtype=torch.float64)
   577:     print(
   578:         "BENCH_DONE "
   579:         f"num_runs={len(results)} mean_steps={float(step_tensor.mean().item()):.6f}",
   580:         flush=True,
   581:     )
   582:     return {
   583:         "config": asdict(config),
   584:         "results": [asdict(result) for result in results],
   585:     }
   586: 
   587: 
   588: def parse_args() -> argparse.Namespace:
   589:     parser = argparse.ArgumentParser(description="Run the MLS-Bench optimization-parity task.")
   590:     parser.add_argument("--seed", type=int, default=42, help="Top-level benchmark seed.")
   591:     parser.add_argument(
   592:         "--output-dir",
   593:         type=Path,
   594:         default=None,
   595:         help="Optional directory for a JSON summary.",
   596:     )
   597:     parser.add_argument(
   598:         "--label",
   599:         type=str,
   600:         default="eval",
   601:         help="Optional label stored in the JSON summary.",
   602:     )
   603:     parser.add_argument(
   604:         "--device",
   605:         choices=("auto", "cpu", "cuda"),
   606:         default="auto",
   607:         help="Execution device.",
   608:     )
   609:     parser.add_argument(
   610:         "--smoke",
   611:         action="store_true",
   612:         help="Run a smaller local sanity check without changing the benchmark defaults in code.",
   613:     )
   614:     parser.add_argument(
   615:         "--n-features",
   616:         type=int,
   617:         default=None,
   618:         help="Override n_features in TaskConfig.",
   619:     )
   620:     parser.add_argument(
   621:         "--secret-size",
   622:         type=int,
   623:         default=None,
   624:         help="Override secret_size in TaskConfig.",
   625:     )
   626:     return parser.parse_args()
   627: 
   628: 
   629: def main() -> None:
   630:     args = parse_args()
   631:     config = maybe_apply_smoke_mode(DEFAULT_TASK, args.smoke)
   632:     if args.n_features is not None:
   633:         config = replace(config, n_features=args.n_features)
   634:     if args.secret_size is not None:
   635:         config = replace(config, secret_size=args.secret_size)
   636:     device = resolve_device(args.device)
   637:     summary = run_benchmark(config=config, seed=args.seed, device=device)
   638: 
   639:     if args.output_dir is not None:
   640:         args.output_dir.mkdir(parents=True, exist_ok=True)
   641:         output_path = args.output_dir / f"{args.label}_seed{args.seed}.json"
   642:         output_path.write_text(json.dumps(summary, indent=2))
   643: 
   644: 
   645: if __name__ == "__main__":
   646:     main()
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
