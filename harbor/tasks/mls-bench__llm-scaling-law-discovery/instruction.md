# MLS-Bench: llm-scaling-law-discovery

# SLDBench Scaling Law Discovery

## Research Question
Design a better scaling-law model that extrapolates on `SLDBench` scaling tasks while keeping a single shared functional form per task and fitting group-specific coefficients from the observed trials. The intended contribution is a compact symbolic law per benchmark family — not generic tabular regression.

## Background
This task is built on the SLDBench benchmark from Liu et al., "Can Language Models Discover Scaling Laws?", 2025, arXiv:2507.21184 (project page: https://linhaowei1.github.io/scaling_law_discovery/). SLDBench collects ~5,000 LLM training experiments from existing scaling-law literature and turns them into symbolic-regression tasks: given numeric experiment descriptors and a categorical group, predict held-out training losses on extrapolation regions.

We use three representative and harder subsets (less saturated than the original `parallel` / `moe` / `sft` trio):

- **`sld-vocab`** — vocabulary scaling law: unigram-normalised loss as a function of non-vocabulary parameters `N`, vocabulary size `V`, and training characters `D`. Reference: Tao et al., "Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies", 2024, arXiv:2407.13623.
- **`sld-lrbsz`** — learning-rate & batch-size scaling law: LM loss as a joint function of learning rate, batch size, training tokens, and non-embedding parameters.
- **`sld-dataconstrained`** — data-constrained scaling law: loss as a function of unique tokens `U`, parameters `N`, and total tokens `D`, where `D` can exceed `U` (data repetition). Reference: Muennighoff et al., "Scaling Data-Constrained Language Models", NeurIPS 2023, arXiv:2305.16264.

## What you can modify
The `ScalingLawModel` class in `custom_scaling_law.py`. Your model receives:

- `X_num` — raw numeric inputs (per-benchmark list below).
- `X_cat` — categorical metadata (primarily the `group`).
- `y` — observed target losses on the training split.

The runtime loads the official `SLDBench` train/test splits from `/data/scaling_law/*.jsonl`. The observed training trials are also mirrored into the editable workspace as read-only files for direct inspection:

- `scaling-law-lab/observed_trials/sld_vocab_train.jsonl`
- `scaling-law-lab/observed_trials/sld_lrbsz_train.jsonl`
- `scaling-law-lab/observed_trials/sld_dataconstrained_train.jsonl`

Inspect these raw trials directly and discover benchmark-specific symbolic laws. Large pretrained LMs are not allowed.

### Benchmarks
- `sld-vocab` — numeric: `non_vocab_parameters`, `vocab_size`, `num_characters`; categorical: `group`; target: `unigram_normalized_loss` (can be negative — do not clip).
- `sld-lrbsz` — numeric: `lr`, `bsz`, `data_size`, `non_embedding_param_size`; categorical: `group`; target: `lm_loss`.
- `sld-dataconstrained` — numeric: `unique_tokens`, `params`, `tokens`; categorical: `group`; target: `loss`.

### Interface
```python
class ScalingLawModel:
    def __init__(self, benchmark_name, numeric_names, categorical_names):
        ...
    def fit(self, X_num, X_cat, y):
        return self
    def predict(self, X_num, X_cat):
        return y_pred
```
`benchmark_name` lets you use different law families for `vocab`, `lrbsz`, and `dataconstrained` while still keeping one shared symbolic expression per benchmark and fitting group-specific coefficients.

## Evaluation
- **Primary**: held-out test `R^2` per benchmark (higher is better).
- **Secondary**: `MAE`, `RMSE`, `NMAE` (lower is better).

Strong solutions usually:
- fit coefficients per `group` rather than collapsing all groups together;
- preserve sensible asymptotics on larger or denser test points (good extrapolation, not memorization).


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scaling-law-lab/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `scaling-law-lab/custom_scaling_law.py`
- editable lines **183–211**


Other files you may **read** for context (do not modify):
- `scaling-law-lab/observed_trials/sld_vocab_train.jsonl`
- `scaling-law-lab/observed_trials/sld_lrbsz_train.jsonl`
- `scaling-law-lab/observed_trials/sld_dataconstrained_train.jsonl`


## Readable Context


### `scaling-law-lab/custom_scaling_law.py`  [EDITABLE — lines 183–211 only]

```python
     1: #!/usr/bin/env python3
     2: """Pure SLDBench scaling-law discovery benchmark."""
     3: 
     4: import argparse
     5: import json
     6: import os
     7: import random
     8: from dataclasses import dataclass
     9: from pathlib import Path
    10: 
    11: import numpy as np
    12: from scipy.optimize import least_squares
    13: 
    14: 
    15: DATA_DIR = Path(os.environ.get("SCALING_LAW_DATA_DIR", "/data/scaling_law"))
    16: EPS = 1e-8
    17: 
    18: 
    19: @dataclass
    20: class BenchmarkData:
    21:     name: str
    22:     X_num_train: np.ndarray
    23:     X_cat_train: np.ndarray
    24:     y_train: np.ndarray
    25:     X_num_test: np.ndarray
    26:     X_cat_test: np.ndarray
    27:     y_test: np.ndarray
    28:     numeric_names: list[str]
    29:     categorical_names: list[str]
    30:     target_name: str
    31: 
    32: 
    33: def set_seed(seed: int) -> None:
    34:     random.seed(seed)
    35:     np.random.seed(seed)
    36: 
    37: 
    38: def load_jsonl(path: Path) -> list[dict]:
    39:     rows = []
    40:     with path.open() as f:
    41:         for line in f:
    42:             line = line.strip()
    43:             if line:
    44:                 rows.append(json.loads(line))
    45:     return rows
    46: 
    47: 
    48: def pick(mapping: dict, keys: list[str], default=None):
    49:     for key in keys:
    50:         if key in mapping and mapping[key] is not None:
    51:             return mapping[key]
    52:     return default
    53: 
    54: 
    55: def safe_float(value, default=np.nan) -> float:
    56:     if value is None:
    57:         return default
    58:     try:
    59:         return float(value)
    60:     except (TypeError, ValueError):
    61:         return default
    62: 
    63: 
    64: def _build_sld_vocab() -> BenchmarkData:
    65:     train_rows = load_jsonl(DATA_DIR / "sldbench__vocab_scaling_law__train.jsonl")
    66:     test_rows = load_jsonl(DATA_DIR / "sldbench__vocab_scaling_law__test.jsonl")
    67: 
    68:     def convert(rows: list[dict]):
    69:         X_num, X_cat, y = [], [], []
    70:         for row in rows:
    71:             X_num.append([
    72:                 safe_float(pick(row, ["non_vocab_parameters", "N"])),
    73:                 safe_float(pick(row, ["vocab_size", "V"])),
    74:                 safe_float(pick(row, ["num_characters", "D"])),
    75:             ])
    76:             X_cat.append([str(pick(row, ["group"], "all_data"))])
    77:             y.append(safe_float(pick(row, ["unigram_normalized_loss", "loss"])))
    78:         return np.asarray(X_num, dtype=float), np.asarray(X_cat, dtype=object), np.asarray(y, dtype=float)
    79: 
    80:     X_num_train, X_cat_train, y_train = convert(train_rows)
    81:     X_num_test, X_cat_test, y_test = convert(test_rows)
    82:     return BenchmarkData(
    83:         name="sld-vocab",
    84:         X_num_train=X_num_train,
    85:         X_cat_train=X_cat_train,
    86:         y_train=y_train,
    87:         X_num_test=X_num_test,
    88:         X_cat_test=X_cat_test,
    89:         y_test=y_test,
    90:         numeric_names=["non_vocab_parameters", "vocab_size", "num_characters"],
    91:         categorical_names=["group"],
    92:         target_name="unigram_normalized_loss",
    93:     )
    94: 
    95: 
    96: def _build_sld_lrbsz() -> BenchmarkData:
    97:     train_rows = load_jsonl(DATA_DIR / "sldbench__lr_bsz_scaling_law__train.jsonl")
    98:     test_rows = load_jsonl(DATA_DIR / "sldbench__lr_bsz_scaling_law__test.jsonl")
    99: 
   100:     def convert(rows: list[dict]):
   101:         X_num, X_cat, y = [], [], []
   102:         for row in rows:
   103:             X_num.append([
   104:                 safe_float(pick(row, ["lr"])),
   105:                 safe_float(pick(row, ["bsz"])),
   106:                 safe_float(pick(row, ["data_size", "D"])),
   107:                 safe_float(pick(row, ["non_embedding_param_size", "N"])),
   108:             ])
   109:             X_cat.append([str(pick(row, ["group"], "all_data"))])
   110:             y.append(safe_float(pick(row, ["lm_loss", "loss"])))
   111:         return np.asarray(X_num, dtype=float), np.asarray(X_cat, dtype=object), np.asarray(y, dtype=float)
   112: 
   113:     X_num_train, X_cat_train, y_train = convert(train_rows)
   114:     X_num_test, X_cat_test, y_test = convert(test_rows)
   115:     return BenchmarkData(
   116:         name="sld-lrbsz",
   117:         X_num_train=X_num_train,
   118:         X_cat_train=X_cat_train,
   119:         y_train=y_train,
   120:         X_num_test=X_num_test,
   121:         X_cat_test=X_cat_test,
   122:         y_test=y_test,
   123:         numeric_names=["lr", "bsz", "data_size", "non_embedding_param_size"],
   124:         categorical_names=["group"],
   125:         target_name="lm_loss",
   126:     )
   127: 
   128: 
   129: def _build_sld_dataconstrained() -> BenchmarkData:
   130:     train_rows = load_jsonl(DATA_DIR / "sldbench__data_constrained_scaling_law__train.jsonl")
   131:     test_rows = load_jsonl(DATA_DIR / "sldbench__data_constrained_scaling_law__test.jsonl")
   132: 
   133:     def convert(rows: list[dict]):
   134:         X_num, X_cat, y = [], [], []
   135:         for row in rows:
   136:             X_num.append([
   137:                 safe_float(pick(row, ["unique_tokens", "U"])),
   138:                 safe_float(pick(row, ["params", "N"])),
   139:                 safe_float(pick(row, ["tokens", "D"])),
   140:             ])
   141:             X_cat.append([str(pick(row, ["group"], "all_data"))])
   142:             y.append(safe_float(pick(row, ["loss"])))
   143:         return np.asarray(X_num, dtype=float), np.asarray(X_cat, dtype=object), np.asarray(y, dtype=float)
   144: 
   145:     X_num_train, X_cat_train, y_train = convert(train_rows)
   146:     X_num_test, X_cat_test, y_test = convert(test_rows)
   147:     return BenchmarkData(
   148:         name="sld-dataconstrained",
   149:         X_num_train=X_num_train,
   150:         X_cat_train=X_cat_train,
   151:         y_train=y_train,
   152:         X_num_test=X_num_test,
   153:         X_cat_test=X_cat_test,
   154:         y_test=y_test,
   155:         numeric_names=["unique_tokens", "params", "tokens"],
   156:         categorical_names=["group"],
   157:         target_name="loss",
   158:     )
   159: 
   160: 
   161: def load_benchmark(name: str) -> BenchmarkData:
   162:     if name == "sld-vocab":
   163:         return _build_sld_vocab()
   164:     if name == "sld-lrbsz":
   165:         return _build_sld_lrbsz()
   166:     if name == "sld-dataconstrained":
   167:         return _build_sld_dataconstrained()
   168:     raise ValueError(f"Unknown benchmark: {name}")
   169: 
   170: 
   171: def group_labels(X_cat: np.ndarray) -> np.ndarray:
   172:     X_cat = np.asarray(X_cat, dtype=object)
   173:     if X_cat.ndim == 1:
   174:         X_cat = X_cat[:, None]
   175:     if X_cat.size == 0 or X_cat.shape[1] == 0:
   176:         return np.asarray(["__all__"] * len(X_cat), dtype=object)
   177:     return np.asarray(
   178:         [str(v) if v is not None else "__MISSING__" for v in X_cat[:, 0]],
   179:         dtype=object,
   180:     )
   181: 
   182: 
   183: # ============================================================
   184: # Scaling Law Model (EDITABLE)
   185: # ============================================================
   186: 
   187: class ScalingLawModel:
   188:     """Editable benchmark-specific symbolic law scaffold.
   189: 
   190:     You may implement different symbolic forms for:
   191:     - sld-vocab
   192:     - sld-lrbsz
   193:     - sld-dataconstrained
   194: 
   195:     The raw observed training trials are mirrored in:
   196:     - observed_trials/sld_vocab_train.jsonl
   197:     - observed_trials/sld_lrbsz_train.jsonl
   198:     - observed_trials/sld_dataconstrained_train.jsonl
   199:     """
   200: 
   201:     def __init__(self, benchmark_name: str, numeric_names=None, categorical_names=None):
   202:         self.benchmark_name = benchmark_name
   203:         self.numeric_names = list(numeric_names or [])
   204:         self.categorical_names = list(categorical_names or [])
   205: 
   206:     def fit(self, X_num, X_cat, y):
   207:         self.mean_ = float(np.mean(y))
   208:         return self
   209: 
   210:     def predict(self, X_num, X_cat):
   211:         return np.full(len(X_num), self.mean_)
   212: 
   213: 
   214: # ============================================================
   215: # Evaluation
   216: # ============================================================
   217: 
   218: def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
   219:     y_true = np.asarray(y_true, dtype=float)
   220:     y_pred = np.asarray(y_pred, dtype=float)
   221:     denom = np.sum((y_true - y_true.mean()) ** 2)
   222:     if denom < EPS:
   223:         return 0.0
   224:     return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)
   225: 
   226: 
   227: def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
   228:     return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))
   229: 
   230: 
   231: def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
   232:     diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
   233:     return float(np.mean(diff ** 2))
   234: 
   235: 
   236: def run_benchmark(benchmark: str, seed: int, output_dir: str) -> None:
   237:     set_seed(seed)
   238:     data = load_benchmark(benchmark)
   239:     model = ScalingLawModel(data.name, data.numeric_names, data.categorical_names)
   240:     model.fit(data.X_num_train, data.X_cat_train, data.y_train)
   241: 
   242:     train_pred = model.predict(data.X_num_train, data.X_cat_train)
   243:     test_pred = model.predict(data.X_num_test, data.X_cat_test)
   244: 
   245:     train_r2 = r2_score(data.y_train, train_pred)
   246:     train_mae = mean_absolute_error(data.y_train, train_pred)
   247:     test_r2 = r2_score(data.y_test, test_pred)
   248:     test_mae = mean_absolute_error(data.y_test, test_pred)
   249:     test_rmse = float(np.sqrt(mean_squared_error(data.y_test, test_pred)))
   250:     test_nmae = float(test_mae / (np.std(data.y_test) + EPS))
   251: 
   252:     n_features = int(getattr(model, "num_features_", data.X_num_train.shape[1]))
   253:     print(
   254:         "TRAIN_METRICS "
   255:         f"n_train={len(data.y_train)} n_test={len(data.y_test)} "
   256:         f"n_features={n_features} train_r2={train_r2:.6f} train_mae={train_mae:.6f}",
   257:         flush=True,
   258:     )
   259:     print(
   260:         "TEST_METRICS "
   261:         f"r2={test_r2:.6f} mae={test_mae:.6f} rmse={test_rmse:.6f} nmae={test_nmae:.6f}",
   262:         flush=True,
   263:     )
   264: 
   265:     output_path = Path(output_dir)
   266:     output_path.mkdir(parents=True, exist_ok=True)
   267:     with (output_path / f"{benchmark}_predictions.json").open("w") as f:
   268:         json.dump(
   269:             {
   270:                 "benchmark": benchmark,
   271:                 "target": data.target_name,
   272:                 "y_true": data.y_test.tolist(),
   273:                 "y_pred": np.asarray(test_pred).tolist(),
   274:                 "metrics": {
   275:                     "r2": test_r2,
   276:                     "mae": float(test_mae),
   277:                     "rmse": test_rmse,
   278:                     "nmae": test_nmae,
   279:                 },
   280:             },
   281:             f,
   282:             indent=2,
   283:         )
   284: 
   285: 
   286: def main():
   287:     parser = argparse.ArgumentParser()
   288:     parser.add_argument("--benchmark", required=True)
   289:     parser.add_argument("--seed", type=int, default=42)
   290:     parser.add_argument("--output-dir", default="./output")
   291:     args = parser.parse_args()
   292:     run_benchmark(args.benchmark, args.seed, args.output_dir)
   293: 
   294: 
   295: if __name__ == "__main__":
   296:     main()
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **sld-vocab** — wall-clock budget `00:15:00`, compute share `1.0`
- **sld-lrbsz** — wall-clock budget `00:15:00`, compute share `1.0`
- **sld-dataconstrained** — wall-clock budget `00:15:00`, compute share `1.0`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `human_exact` baseline — editable region  [READ-ONLY — reference implementation]

In `scaling-law-lab/custom_scaling_law.py`:

```python
Lines 183–442:
   180:     )
   181: 
   182: 
   183: def _safe_log_residuals(pred, y):
   184:     pred = np.clip(np.asarray(pred, dtype=float), EPS, None)
   185:     y = np.clip(np.asarray(y, dtype=float), EPS, None)
   186:     return np.log(pred) - np.log(y)
   187: 
   188: 
   189: def _linear_residuals(pred, y):
   190:     pred = np.asarray(pred, dtype=float)
   191:     y = np.asarray(y, dtype=float)
   192:     return pred - y
   193: 
   194: 
   195: def _fit_generic(X, y, init_u, unpack_fn, predict_fn, n_restarts=6,
   196:                  use_log=True):
   197:     init_u = np.asarray(init_u, dtype=float)
   198:     y = np.asarray(y, dtype=float)
   199:     rng = np.random.default_rng(np.random.randint(0, 2**32 - 1))
   200:     candidates = [init_u]
   201:     for scale in np.linspace(0.05, 0.45, max(n_restarts - 1, 0)):
   202:         candidates.append(init_u + rng.normal(scale=scale, size=init_u.shape))
   203:     best_u, best_score = init_u, float("inf")
   204:     for u0 in candidates:
   205:         def residuals(u):
   206:             try:
   207:                 pred = predict_fn(X, unpack_fn(u))
   208:                 resid = (_safe_log_residuals(pred, y) if use_log
   209:                          else _linear_residuals(pred, y))
   210:                 return np.nan_to_num(resid, nan=1e3, posinf=1e3, neginf=-1e3)
   211:             except Exception:
   212:                 return np.full_like(y, 1e3, dtype=float)
   213:         try:
   214:             result = least_squares(residuals, u0, method="trf",
   215:                                    loss="soft_l1", f_scale=0.05, max_nfev=5000)
   216:             u_opt = result.x
   217:         except Exception:
   218:             u_opt = np.asarray(u0, dtype=float)
   219:         pred = predict_fn(X, unpack_fn(u_opt))
   220:         if use_log:
   221:             score = float(np.mean(_safe_log_residuals(pred, y) ** 2))
   222:         else:
   223:             score = float(np.mean((np.asarray(pred, dtype=float) - y) ** 2))
   224:         if np.isfinite(score) and score < best_score:
   225:             best_score, best_u = score, u_opt
   226:     return unpack_fn(best_u)
   227: 
   228: 
   229: # -------- sld-vocab: L = E + A*N^-alpha + B*V^-beta + C*D^-gamma --------
   230: 
   231: def _vocab_human_predict(X, params):
   232:     n = np.clip(np.asarray(X[:, 0], dtype=float), 1.0, None)
   233:     v = np.clip(np.asarray(X[:, 1], dtype=float), 1.0, None)
   234:     d = np.clip(np.asarray(X[:, 2], dtype=float), 1.0, None)
   235:     E, A, alpha, B, beta, C, gamma = params
   236:     return (E
   237:             + A * np.power(n, -alpha)
   238:             + B * np.power(v, -beta)
   239:             + C * np.power(d, -gamma))
   240: 
   241: 
   242: def _fit_vocab_human(X, y):
   243:     y = np.asarray(y, dtype=float)
   244:     def unpack(u):
   245:         # E unconstrained; scale / exponent parameters exponentiated.
   246:         E = u[0]
   247:         A = np.exp(u[1])
   248:         alpha = np.exp(u[2])
   249:         B = np.exp(u[3])
   250:         beta = np.exp(u[4])
   251:         C = np.exp(u[5])
   252:         gamma = np.exp(u[6])
   253:         return np.array([E, A, alpha, B, beta, C, gamma], dtype=float)
   254:     init = np.array([
   255:         float(np.median(y)),
   256:         np.log(max(abs(np.std(y)), 0.1)), np.log(0.1),
   257:         np.log(max(abs(np.std(y)), 0.1)), np.log(0.3),
   258:         np.log(max(abs(np.std(y)), 0.1)), np.log(0.3),
   259:     ])
   260:     return _fit_generic(X, y, init, unpack, _vocab_human_predict,
   261:                         n_restarts=8, use_log=False)
   262: 
   263: 
   264: # -------- sld-lrbsz: Expert-B human law from SLDBench paper --------
   265: # L(D, N, l, b) = A/D^alpha + B/N^beta + C + K_l*(l - l0)^2 + E*(log b + b0/b)
   266: # with l0 = F * N^gamma * D^zeta, b0 = G * D^eta.
   267: # Reference: arXiv:2507.21184v5 Appendix A.4 (Expert B law; R^2 = -0.0756).
   268: # Code parameter K_l is named D_lr in the paper's reference implementation.
   269: 
   270: def _lrbsz_human_predict(X, params):
   271:     lr = np.clip(np.asarray(X[:, 0], dtype=float), 1e-12, None)
   272:     bsz = np.clip(np.asarray(X[:, 1], dtype=float), 1e-12, None)
   273:     d = np.clip(np.asarray(X[:, 2], dtype=float), 1.0, None)
   274:     n = np.clip(np.asarray(X[:, 3], dtype=float), 1.0, None)
   275:     A, alpha, B, beta, C, D_lr, E, F, gamma, zeta, G, eta = params
   276:     l0 = F * np.power(n, gamma) * np.power(d, zeta)
   277:     b0 = G * np.power(d, eta)
   278:     term_data = A * np.power(d, -alpha)
   279:     term_param = B * np.power(n, -beta)
   280:     term_lr = D_lr * (lr - l0) ** 2
   281:     term_bsz = E * (np.log(bsz) + b0 / bsz)
   282:     return term_data + term_param + C + term_lr + term_bsz
   283: 
   284: 
   285: def _fit_lrbsz_human(X, y):
   286:     y = np.asarray(y, dtype=float)
   287: 
   288:     # Reference coefficients from the SLDBench paper (Expert B, "all_data"):
   289:     #   [A, alpha, B, beta, C, D_lr, E, F, gamma, zeta, G, eta]
   290:     paper_params = np.array([
   291:         262.1391, 0.2675, 7.0285, 0.0746, 0.0000136, 1278.595,
   292:         0.0493, 0.3242, -1.0580, 0.6498, 0.0302, 0.3503,
   293:     ], dtype=float)
   294: 
   295:     # Parameterise so the fitter explores physically meaningful regions while
   296:     # remaining well-conditioned. Positive quantities are exponentiated;
   297:     # signed exponents (gamma, zeta, eta) are unconstrained.
   298:     def unpack(u):
   299:         A = np.exp(u[0]); alpha = np.exp(u[1])
   300:         B = np.exp(u[2]); beta = np.exp(u[3])
   301:         C = u[4]
   302:         D_lr = np.exp(u[5])
   303:         E = np.exp(u[6])
   304:         F = np.exp(u[7]); gamma = u[8]; zeta = u[9]
   305:         G = np.exp(u[10]); eta = u[11]
   306:         return np.array([A, alpha, B, beta, C, D_lr, E, F, gamma, zeta, G, eta],
   307:                         dtype=float)
   308: 
   309:     def pack(p):
   310:         A, alpha, B, beta, C, D_lr, E, F, gamma, zeta, G, eta = p
   311:         return np.array([
   312:             np.log(max(A, 1e-12)), np.log(max(alpha, 1e-12)),
   313:             np.log(max(B, 1e-12)), np.log(max(beta, 1e-12)),
   314:             C,
   315:             np.log(max(D_lr, 1e-12)),
   316:             np.log(max(E, 1e-12)),
   317:             np.log(max(F, 1e-12)), gamma, zeta,
   318:             np.log(max(G, 1e-12)), eta,
   319:         ], dtype=float)
   320: 
   321:     init_paper = pack(paper_params)
   322: 
   323:     # Also include a data-driven init so we degrade gracefully if the training
   324:     # split shifts the optimum.
   325:     y_span = max(float(y.max() - y.min()), 0.1)
   326:     init_data = np.array([
   327:         np.log(max(y_span, 0.1)), np.log(0.25),
   328:         np.log(max(y_span, 0.1)), np.log(0.1),
   329:         float(max(y.min(), 0.01)),
   330:         np.log(1e3), np.log(0.05),
   331:         np.log(0.3), -1.0, 0.65,
   332:         np.log(0.03), 0.35,
   333:     ], dtype=float)
   334: 
   335:     # Evaluate the reference coefficients directly (no fit) as an absolute
   336:     # fallback — they already achieve the reported R^2 = -0.0756.
   337:     best_params = paper_params
   338:     best_score = float(np.mean((_lrbsz_human_predict(X, paper_params) - y) ** 2))
   339:     if not np.isfinite(best_score):
   340:         best_score = float("inf")
   341: 
   342:     for u0 in (init_paper, init_data):
   343:         params = _fit_generic(X, y, u0, unpack, _lrbsz_human_predict,
   344:                               n_restarts=3, use_log=False)
   345:         pred = _lrbsz_human_predict(X, params)
   346:         score = float(np.mean((pred - y) ** 2))
   347:         if np.isfinite(score) and score < best_score:
   348:             best_score, best_params = score, params
   349:     return best_params
   350: 
   351: 
   352: # -------- sld-dataconstrained: Muennighoff et al. with effective tokens --
   353: 
   354: def _dconstrained_human_predict(X, params):
   355:     u = np.clip(np.asarray(X[:, 0], dtype=float), 1.0, None)   # unique_tokens
   356:     n = np.clip(np.asarray(X[:, 1], dtype=float), 1.0, None)   # params
   357:     d = np.clip(np.asarray(X[:, 2], dtype=float), 1.0, None)   # tokens
   358:     E, A, alpha, B, beta = params
   359:     # Effective tokens: U * (1 - exp(-D/U)) saturates when D >> U (repeated data).
   360:     d_eff = u * (1.0 - np.exp(-np.clip(d / u, 0.0, 50.0)))
   361:     d_eff = np.maximum(d_eff, 1.0)
   362:     return E + A * np.power(n, -alpha) + B * np.power(d_eff, -beta)
   363: 
   364: 
   365: def _fit_dconstrained_human(X, y):
   366:     y = np.asarray(y, dtype=float)
   367:     def unpack(u):
   368:         E = np.exp(u[0])
   369:         A = np.exp(u[1]); alpha = np.exp(u[2])
   370:         B = np.exp(u[3]); beta = np.exp(u[4])
   371:         return np.array([E, A, alpha, B, beta], dtype=float)
   372:     init = np.array([
   373:         np.log(max(y.min() * 0.9, 0.1)),
   374:         np.log(max(y.max() - y.min(), 0.1)), np.log(0.35),
   375:         np.log(max(y.max() - y.min(), 0.1)), np.log(0.35),
   376:     ])
   377:     return _fit_generic(X, y, init, unpack, _dconstrained_human_predict,
   378:                         n_restarts=8, use_log=True)
   379: 
   380: 
   381: def _human_fit_params(benchmark_name, X, y):
   382:     if benchmark_name == "sld-vocab":
   383:         return _fit_vocab_human(X, y)
   384:     if benchmark_name == "sld-lrbsz":
   385:         return _fit_lrbsz_human(X, y)
   386:     if benchmark_name == "sld-dataconstrained":
   387:         return _fit_dconstrained_human(X, y)
   388:     raise ValueError(f"Unsupported benchmark: {benchmark_name}")
   389: 
   390: 
   391: def _human_predict_params(benchmark_name, X, params):
   392:     if benchmark_name == "sld-vocab":
   393:         return _vocab_human_predict(X, params)
   394:     if benchmark_name == "sld-lrbsz":
   395:         return _lrbsz_human_predict(X, params)
   396:     if benchmark_name == "sld-dataconstrained":
   397:         return _dconstrained_human_predict(X, params)
   398:     raise ValueError(f"Unsupported benchmark: {benchmark_name}")
   399: 
   400: 
   401: class ScalingLawModel:
   402:     """Human law family from the literature for the harder SLDBench subsets.
   403: 
   404:     Benchmark-specific symbolic forms, fit per group via nonlinear least
   405:     squares:
   406:     - vocab: additive Chinchilla-style with per-axis power terms
   407:     - lrbsz: SLDBench Expert-B hierarchical additive law (arXiv:2507.21184)
   408:     - dataconstrained: Muennighoff-style effective-token saturation
   409:     """
   410: 
   411:     def __init__(self, benchmark_name, numeric_names=None, categorical_names=None):
   412:         self.benchmark_name = benchmark_name
   413:         self.numeric_names = list(numeric_names or [])
   414:         self.categorical_names = list(categorical_names or [])
   415:         self.group_params_ = {}
   416:         self.default_params_ = None
   417: 
   418:     def fit(self, X_num, X_cat, y):
   419:         X_num = np.asarray(X_num, dtype=float)
   420:         y = np.asarray(y, dtype=float)
   421:         labels = group_labels(X_cat)
   422:         fitted = []
   423:         for group in sorted(set(labels.tolist())):
   424:             mask = labels == group
   425:             params = _human_fit_params(self.benchmark_name, X_num[mask], y[mask])
   426:             self.group_params_[group] = params
   427:             fitted.append(params)
   428:         self.default_params_ = np.median(np.stack(fitted, axis=0), axis=0)
   429:         return self
   430: 
   431:     def predict(self, X_num, X_cat):
   432:         X_num = np.asarray(X_num, dtype=float)
   433:         labels = group_labels(X_cat)
   434:         preds = np.zeros(len(labels), dtype=float)
   435:         for group in sorted(set(labels.tolist())):
   436:             mask = labels == group
   437:             params = self.group_params_.get(group, self.default_params_)
   438:             preds[mask] = _human_predict_params(self.benchmark_name,
   439:                                                 X_num[mask], params)
   440:         # Do not clip to positive: vocab target (unigram_normalized_loss) can
   441:         # be negative.
   442:         return preds
   443: 
   444: 
   445: # ============================================================
```

### `sldagent_style` baseline — editable region  [READ-ONLY — reference implementation]

In `scaling-law-lab/custom_scaling_law.py`:

```python
Lines 183–391:
   180:     )
   181: 
   182: 
   183: def _safe_log_residuals(pred, y):
   184:     pred = np.clip(np.asarray(pred, dtype=float), EPS, None)
   185:     y = np.clip(np.asarray(y, dtype=float), EPS, None)
   186:     return np.log(pred) - np.log(y)
   187: 
   188: 
   189: def _linear_residuals(pred, y):
   190:     return np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)
   191: 
   192: 
   193: def _fit_generic(X, y, init_u, unpack_fn, predict_fn, n_restarts=6,
   194:                  use_log=True):
   195:     init_u = np.asarray(init_u, dtype=float)
   196:     y = np.asarray(y, dtype=float)
   197:     rng = np.random.default_rng(np.random.randint(0, 2**32 - 1))
   198:     candidates = [init_u]
   199:     for scale in np.linspace(0.05, 0.45, max(n_restarts - 1, 0)):
   200:         candidates.append(init_u + rng.normal(scale=scale, size=init_u.shape))
   201:     best_u, best_score = init_u, float("inf")
   202:     for u0 in candidates:
   203:         def residuals(u):
   204:             try:
   205:                 pred = predict_fn(X, unpack_fn(u))
   206:                 resid = (_safe_log_residuals(pred, y) if use_log
   207:                          else _linear_residuals(pred, y))
   208:                 return np.nan_to_num(resid, nan=1e3, posinf=1e3, neginf=-1e3)
   209:             except Exception:
   210:                 return np.full_like(y, 1e3, dtype=float)
   211:         try:
   212:             result = least_squares(residuals, u0, method="trf",
   213:                                    loss="soft_l1", f_scale=0.05, max_nfev=5000)
   214:             u_opt = result.x
   215:         except Exception:
   216:             u_opt = np.asarray(u0, dtype=float)
   217:         pred = predict_fn(X, unpack_fn(u_opt))
   218:         if use_log:
   219:             score = float(np.mean(_safe_log_residuals(pred, y) ** 2))
   220:         else:
   221:             score = float(np.mean((np.asarray(pred, dtype=float) - y) ** 2))
   222:         if np.isfinite(score) and score < best_score:
   223:             best_score, best_u = score, u_opt
   224:     return unpack_fn(best_u)
   225: 
   226: 
   227: # -------- sld-vocab: multiplicative interaction on log scales --------
   228: 
   229: def _vocab_sldagent_predict(X, params):
   230:     n = np.clip(np.asarray(X[:, 0], dtype=float), 1.0, None)
   231:     v = np.clip(np.asarray(X[:, 1], dtype=float), 1.0, None)
   232:     d = np.clip(np.asarray(X[:, 2], dtype=float), 1.0, None)
   233:     E, A, a1, a2, a3, A_vd, g1, g2 = params
   234:     # Cross term links vocab and data.
   235:     cross = A_vd * np.power(v, -g1) * np.power(d, -g2)
   236:     return E + A * np.power(n, -a1) * np.power(v, -a2) * np.power(d, -a3) + cross
   237: 
   238: 
   239: def _fit_vocab_sldagent(X, y):
   240:     y = np.asarray(y, dtype=float)
   241:     def unpack(u):
   242:         E = u[0]
   243:         A = np.exp(u[1])
   244:         a1, a2, a3 = np.exp(u[2]), np.exp(u[3]), np.exp(u[4])
   245:         A_vd = np.exp(u[5])
   246:         g1, g2 = np.exp(u[6]), np.exp(u[7])
   247:         return np.array([E, A, a1, a2, a3, A_vd, g1, g2], dtype=float)
   248:     init = np.array([
   249:         float(np.median(y)),
   250:         np.log(max(abs(np.std(y)), 0.1)),
   251:         np.log(0.1), np.log(0.2), np.log(0.2),
   252:         np.log(max(abs(np.std(y)), 0.05)),
   253:         np.log(0.3), np.log(0.3),
   254:     ])
   255:     return _fit_generic(X, y, init, unpack, _vocab_sldagent_predict,
   256:                         n_restarts=8, use_log=False)
   257: 
   258: 
   259: # -------- sld-lrbsz: Chinchilla base + joint (lr, bsz) coupling --------
   260: 
   261: def _lrbsz_sldagent_predict(X, params):
   262:     lr = np.clip(np.asarray(X[:, 0], dtype=float), 1e-8, None)
   263:     bsz = np.clip(np.asarray(X[:, 1], dtype=float), 1.0, None)
   264:     d = np.clip(np.asarray(X[:, 2], dtype=float), 1.0, None)
   265:     n = np.clip(np.asarray(X[:, 3], dtype=float), 1.0, None)
   266:     E, A, alpha, B, beta, k, log_lr_star, log_bsz_star, rho = params
   267:     base = E + A * np.power(n, -alpha) + B * np.power(d, -beta)
   268:     dx = np.log(lr) - log_lr_star
   269:     dy = np.log(bsz) - log_bsz_star
   270:     # Correlated quadratic bowl around (lr*, bsz*) with coupling rho.
   271:     penalty = k * (dx * dx + dy * dy + 2.0 * rho * dx * dy)
   272:     return base + penalty
   273: 
   274: 
   275: def _fit_lrbsz_sldagent(X, y):
   276:     y = np.asarray(y, dtype=float)
   277:     lr = np.clip(np.asarray(X[:, 0], dtype=float), 1e-8, None)
   278:     bsz = np.clip(np.asarray(X[:, 1], dtype=float), 1.0, None)
   279:     def unpack(u):
   280:         E = u[0]
   281:         A = np.exp(u[1]); alpha = np.exp(u[2])
   282:         B = np.exp(u[3]); beta = np.exp(u[4])
   283:         k = np.exp(u[5])
   284:         log_lr_star = u[6]; log_bsz_star = u[7]
   285:         rho = np.tanh(u[8])  # keep in (-1, 1)
   286:         return np.array([E, A, alpha, B, beta, k,
   287:                          log_lr_star, log_bsz_star, rho], dtype=float)
   288:     init = np.array([
   289:         float(max(y.min() * 0.9, 0.1)),
   290:         np.log(max(y.max() - y.min(), 0.1)), np.log(0.3),
   291:         np.log(max(y.max() - y.min(), 0.1)), np.log(0.3),
   292:         np.log(0.05),
   293:         float(np.log(np.median(lr))), float(np.log(np.median(bsz))),
   294:         0.0,
   295:     ])
   296:     return _fit_generic(X, y, init, unpack, _lrbsz_sldagent_predict,
   297:                         n_restarts=10, use_log=True)
   298: 
   299: 
   300: # -------- sld-dataconstrained: multiplicative repeat-efficiency term ---
   301: 
   302: def _dconstrained_sldagent_predict(X, params):
   303:     u = np.clip(np.asarray(X[:, 0], dtype=float), 1.0, None)
   304:     n = np.clip(np.asarray(X[:, 1], dtype=float), 1.0, None)
   305:     d = np.clip(np.asarray(X[:, 2], dtype=float), 1.0, None)
   306:     E, A, alpha, B, beta, R = params
   307:     ratio = np.clip(d / u, 0.0, 200.0)
   308:     # Repeat-efficiency: multiplier decays smoothly with repetition.
   309:     efficiency = 1.0 / (1.0 + ratio / np.maximum(R, 1e-3))
   310:     d_eff = np.maximum(d * efficiency, 1.0)
   311:     return E + A * np.power(n, -alpha) + B * np.power(d_eff, -beta)
   312: 
   313: 
   314: def _fit_dconstrained_sldagent(X, y):
   315:     y = np.asarray(y, dtype=float)
   316:     def unpack(u):
   317:         E = np.exp(u[0])
   318:         A = np.exp(u[1]); alpha = np.exp(u[2])
   319:         B = np.exp(u[3]); beta = np.exp(u[4])
   320:         R = np.exp(u[5])
   321:         return np.array([E, A, alpha, B, beta, R], dtype=float)
   322:     init = np.array([
   323:         np.log(max(y.min() * 0.9, 0.1)),
   324:         np.log(max(y.max() - y.min(), 0.1)), np.log(0.35),
   325:         np.log(max(y.max() - y.min(), 0.1)), np.log(0.35),
   326:         np.log(5.0),
   327:     ])
   328:     return _fit_generic(X, y, init, unpack, _dconstrained_sldagent_predict,
   329:                         n_restarts=8, use_log=True)
   330: 
   331: 
   332: def _sldagent_fit_params(benchmark_name, X, y):
   333:     if benchmark_name == "sld-vocab":
   334:         return _fit_vocab_sldagent(X, y)
   335:     if benchmark_name == "sld-lrbsz":
   336:         return _fit_lrbsz_sldagent(X, y)
   337:     if benchmark_name == "sld-dataconstrained":
   338:         return _fit_dconstrained_sldagent(X, y)
   339:     raise ValueError(f"Unsupported benchmark: {benchmark_name}")
   340: 
   341: 
   342: def _sldagent_predict_params(benchmark_name, X, params):
   343:     if benchmark_name == "sld-vocab":
   344:         return _vocab_sldagent_predict(X, params)
   345:     if benchmark_name == "sld-lrbsz":
   346:         return _lrbsz_sldagent_predict(X, params)
   347:     if benchmark_name == "sld-dataconstrained":
   348:         return _dconstrained_sldagent_predict(X, params)
   349:     raise ValueError(f"Unsupported benchmark: {benchmark_name}")
   350: 
   351: 
   352: class ScalingLawModel:
   353:     """SLDAgent-style symbolic baseline for the harder SLDBench subsets.
   354: 
   355:     Uses discovered-style symbolic forms with cross-axis interactions:
   356:     - vocab: additive power law with extra V*D cross term
   357:     - lrbsz: Chinchilla base + correlated (lr, bsz) quadratic bowl
   358:     - dataconstrained: multiplicative repeat-efficiency factor on D_eff
   359:     """
   360: 
   361:     def __init__(self, benchmark_name, numeric_names=None, categorical_names=None):
   362:         self.benchmark_name = benchmark_name
   363:         self.numeric_names = list(numeric_names or [])
   364:         self.categorical_names = list(categorical_names or [])
   365:         self.group_params_ = {}
   366:         self.default_params_ = None
   367: 
   368:     def fit(self, X_num, X_cat, y):
   369:         X_num = np.asarray(X_num, dtype=float)
   370:         y = np.asarray(y, dtype=float)
   371:         labels = group_labels(X_cat)
   372:         fitted = []
   373:         for group in sorted(set(labels.tolist())):
   374:             mask = labels == group
   375:             params = _sldagent_fit_params(self.benchmark_name,
   376:                                           X_num[mask], y[mask])
   377:             self.group_params_[group] = params
   378:             fitted.append(params)
   379:         self.default_params_ = np.median(np.stack(fitted, axis=0), axis=0)
   380:         return self
   381: 
   382:     def predict(self, X_num, X_cat):
   383:         X_num = np.asarray(X_num, dtype=float)
   384:         labels = group_labels(X_cat)
   385:         preds = np.zeros(len(labels), dtype=float)
   386:         for group in sorted(set(labels.tolist())):
   387:             mask = labels == group
   388:             params = self.group_params_.get(group, self.default_params_)
   389:             preds[mask] = _sldagent_predict_params(self.benchmark_name,
   390:                                                    X_num[mask], params)
   391:         return preds
   392: 
   393: 
   394: # ============================================================
```

### `kernel_ridge` baseline — editable region  [READ-ONLY — reference implementation]

In `scaling-law-lab/custom_scaling_law.py`:

```python
Lines 183–276:
   180:     )
   181: 
   182: 
   183: from sklearn.kernel_ridge import KernelRidge as _KernelRidge
   184: 
   185: 
   186: class _FeatureMap:
   187:     """Mixed numeric/categorical encoder for black-box baselines."""
   188: 
   189:     def __init__(self, include_raw=True, include_log=True):
   190:         self.include_raw = include_raw
   191:         self.include_log = include_log
   192: 
   193:     def fit(self, X_num, X_cat):
   194:         X_num = np.asarray(X_num, dtype=float)
   195:         self.num_medians_ = np.nanmedian(X_num, axis=0)
   196:         self.num_medians_ = np.where(np.isnan(self.num_medians_), 0.0,
   197:                                      self.num_medians_)
   198:         filled = np.where(np.isnan(X_num), self.num_medians_, X_num)
   199:         self.raw_mean_ = filled.mean(axis=0)
   200:         self.raw_std_ = filled.std(axis=0)
   201:         self.raw_std_[self.raw_std_ < 1e-8] = 1.0
   202:         clipped = np.clip(filled, a_min=0.0, a_max=None)
   203:         logged = np.log1p(clipped)
   204:         self.log_mean_ = logged.mean(axis=0)
   205:         self.log_std_ = logged.std(axis=0)
   206:         self.log_std_[self.log_std_ < 1e-8] = 1.0
   207:         self.cat_levels_ = []
   208:         X_cat = np.asarray(X_cat, dtype=object)
   209:         for col in range(X_cat.shape[1]):
   210:             values = [str(v) if v is not None else "__MISSING__"
   211:                       for v in X_cat[:, col]]
   212:             self.cat_levels_.append(sorted(set(values)))
   213:         return self
   214: 
   215:     def _transform_num(self, X_num):
   216:         X_num = np.asarray(X_num, dtype=float)
   217:         filled = np.where(np.isnan(X_num), self.num_medians_, X_num)
   218:         pieces = []
   219:         if self.include_raw:
   220:             pieces.append((filled - self.raw_mean_) / self.raw_std_)
   221:         if self.include_log:
   222:             logged = np.log1p(np.clip(filled, a_min=0.0, a_max=None))
   223:             pieces.append((logged - self.log_mean_) / self.log_std_)
   224:         return np.concatenate(pieces, axis=1) if pieces else filled
   225: 
   226:     def _transform_cat(self, X_cat):
   227:         X_cat = np.asarray(X_cat, dtype=object)
   228:         if X_cat.shape[1] == 0:
   229:             return np.empty((X_cat.shape[0], 0), dtype=float)
   230:         cols = []
   231:         for col, levels in enumerate(self.cat_levels_):
   232:             values = [str(v) if v is not None else "__MISSING__"
   233:                       for v in X_cat[:, col]]
   234:             onehot = np.zeros((X_cat.shape[0], len(levels)), dtype=float)
   235:             level_to_idx = {level: idx for idx, level in enumerate(levels)}
   236:             for row_idx, value in enumerate(values):
   237:                 idx = level_to_idx.get(value)
   238:                 if idx is not None:
   239:                     onehot[row_idx, idx] = 1.0
   240:             cols.append(onehot)
   241:         return np.concatenate(cols, axis=1)
   242: 
   243:     def transform(self, X_num, X_cat):
   244:         num = self._transform_num(X_num)
   245:         cat = self._transform_cat(X_cat)
   246:         if cat.size == 0:
   247:             return num
   248:         if num.size == 0:
   249:             return cat
   250:         return np.concatenate([num, cat], axis=1)
   251: 
   252:     def fit_transform(self, X_num, X_cat):
   253:         return self.fit(X_num, X_cat).transform(X_num, X_cat)
   254: 
   255: 
   256: class ScalingLawModel:
   257:     """Black-box kernel ridge baseline on mixed SLDBench features."""
   258: 
   259:     def __init__(self, benchmark_name, numeric_names=None,
   260:                  categorical_names=None):
   261:         self.benchmark_name = benchmark_name
   262:         self.encoder = _FeatureMap(include_raw=True, include_log=True)
   263:         self.model = None
   264:         self.num_features_ = 0
   265: 
   266:     def fit(self, X_num, X_cat, y):
   267:         features = self.encoder.fit_transform(X_num, X_cat)
   268:         gamma = 1.0 / max(features.shape[1], 1)
   269:         self.model = _KernelRidge(alpha=0.05, kernel="rbf", gamma=gamma)
   270:         self.model.fit(features, np.asarray(y, dtype=float))
   271:         self.num_features_ = features.shape[1]
   272:         return self
   273: 
   274:     def predict(self, X_num, X_cat):
   275:         features = self.encoder.transform(X_num, X_cat)
   276:         return np.asarray(self.model.predict(features), dtype=float)
   277: 
   278: 
   279: # ============================================================
```

### `xgboost` baseline — editable region  [READ-ONLY — reference implementation]

In `scaling-law-lab/custom_scaling_law.py`:

```python
Lines 183–298:
   180:     )
   181: 
   182: 
   183: try:
   184:     from xgboost import XGBRegressor as _XGBRegressor
   185: except Exception:
   186:     _XGBRegressor = None
   187: from sklearn.ensemble import GradientBoostingRegressor as _GBR
   188: 
   189: 
   190: class _FeatureMap:
   191:     """Mixed numeric/categorical encoder for black-box baselines."""
   192: 
   193:     def __init__(self, include_raw=True, include_log=True):
   194:         self.include_raw = include_raw
   195:         self.include_log = include_log
   196: 
   197:     def fit(self, X_num, X_cat):
   198:         X_num = np.asarray(X_num, dtype=float)
   199:         self.num_medians_ = np.nanmedian(X_num, axis=0)
   200:         self.num_medians_ = np.where(np.isnan(self.num_medians_), 0.0,
   201:                                      self.num_medians_)
   202:         filled = np.where(np.isnan(X_num), self.num_medians_, X_num)
   203:         self.raw_mean_ = filled.mean(axis=0)
   204:         self.raw_std_ = filled.std(axis=0)
   205:         self.raw_std_[self.raw_std_ < 1e-8] = 1.0
   206:         clipped = np.clip(filled, a_min=0.0, a_max=None)
   207:         logged = np.log1p(clipped)
   208:         self.log_mean_ = logged.mean(axis=0)
   209:         self.log_std_ = logged.std(axis=0)
   210:         self.log_std_[self.log_std_ < 1e-8] = 1.0
   211:         self.cat_levels_ = []
   212:         X_cat = np.asarray(X_cat, dtype=object)
   213:         for col in range(X_cat.shape[1]):
   214:             values = [str(v) if v is not None else "__MISSING__"
   215:                       for v in X_cat[:, col]]
   216:             self.cat_levels_.append(sorted(set(values)))
   217:         return self
   218: 
   219:     def _transform_num(self, X_num):
   220:         X_num = np.asarray(X_num, dtype=float)
   221:         filled = np.where(np.isnan(X_num), self.num_medians_, X_num)
   222:         pieces = []
   223:         if self.include_raw:
   224:             pieces.append((filled - self.raw_mean_) / self.raw_std_)
   225:         if self.include_log:
   226:             logged = np.log1p(np.clip(filled, a_min=0.0, a_max=None))
   227:             pieces.append((logged - self.log_mean_) / self.log_std_)
   228:         return np.concatenate(pieces, axis=1) if pieces else filled
   229: 
   230:     def _transform_cat(self, X_cat):
   231:         X_cat = np.asarray(X_cat, dtype=object)
   232:         if X_cat.shape[1] == 0:
   233:             return np.empty((X_cat.shape[0], 0), dtype=float)
   234:         cols = []
   235:         for col, levels in enumerate(self.cat_levels_):
   236:             values = [str(v) if v is not None else "__MISSING__"
   237:                       for v in X_cat[:, col]]
   238:             onehot = np.zeros((X_cat.shape[0], len(levels)), dtype=float)
   239:             level_to_idx = {level: idx for idx, level in enumerate(levels)}
   240:             for row_idx, value in enumerate(values):
   241:                 idx = level_to_idx.get(value)
   242:                 if idx is not None:
   243:                     onehot[row_idx, idx] = 1.0
   244:             cols.append(onehot)
   245:         return np.concatenate(cols, axis=1)
   246: 
   247:     def transform(self, X_num, X_cat):
   248:         num = self._transform_num(X_num)
   249:         cat = self._transform_cat(X_cat)
   250:         if cat.size == 0:
   251:             return num
   252:         if num.size == 0:
   253:             return cat
   254:         return np.concatenate([num, cat], axis=1)
   255: 
   256:     def fit_transform(self, X_num, X_cat):
   257:         return self.fit(X_num, X_cat).transform(X_num, X_cat)
   258: 
   259: 
   260: class ScalingLawModel:
   261:     """Boosted-tree baseline on mixed SLDBench features."""
   262: 
   263:     def __init__(self, benchmark_name, numeric_names=None,
   264:                  categorical_names=None):
   265:         self.benchmark_name = benchmark_name
   266:         self.encoder = _FeatureMap(include_raw=True, include_log=True)
   267:         self.model = None
   268:         self.num_features_ = 0
   269: 
   270:     def fit(self, X_num, X_cat, y):
   271:         seed = int(os.environ.get("SEED", "42"))
   272:         features = self.encoder.fit_transform(X_num, X_cat)
   273:         y_arr = np.asarray(y, dtype=float)
   274:         # Fit in log-space only when the target is strictly positive (e.g.
   275:         # lm_loss, training loss). Targets like vocab unigram-normalised loss
   276:         # can be negative, so fall back to linear regression in that case.
   277:         self._use_log = bool(np.all(y_arr > 0.0))
   278:         target = np.log(np.clip(y_arr, EPS, None)) if self._use_log else y_arr
   279:         if _XGBRegressor is not None:
   280:             self.model = _XGBRegressor(
   281:                 objective="reg:squarederror",
   282:                 n_estimators=120, max_depth=3, learning_rate=0.05,
   283:                 subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
   284:                 tree_method="hist", n_jobs=4, verbosity=0, random_state=seed,
   285:             )
   286:         else:
   287:             self.model = _GBR(
   288:                 n_estimators=120, learning_rate=0.05, max_depth=3,
   289:                 random_state=seed,
   290:             )
   291:         self.model.fit(features, target)
   292:         self.num_features_ = features.shape[1]
   293:         return self
   294: 
   295:     def predict(self, X_num, X_cat):
   296:         features = self.encoder.transform(X_num, X_cat)
   297:         raw = np.asarray(self.model.predict(features), dtype=float)
   298:         return np.exp(raw) if getattr(self, "_use_log", False) else raw
   299: 
   300: 
   301: # ============================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
