# MLS-Bench: ml-selective-deferral

# Selective Deferral Under Subgroup Shift

## Research Question
Design a selective-prediction / deferral policy for high-stakes tabular decisions. The base classifier and train/calibration/test pipeline are fixed; the contribution is the *acceptance rule* that decides — given a target coverage — which test examples are accepted and which are deferred to a downstream reviewer or backup process.

A good policy should:
- keep selective risk low at the target coverage,
- avoid concentrating deferrals on one subgroup,
- preserve AUROC of its acceptance score as a confidence-ranking signal,
- be simple enough to fit and apply offline on modest compute.

## Background
Selective prediction lets a fixed classifier abstain when its prediction is unreliable. Under subgroup shift, naive confidence thresholds can defer disproportionately on one group while leaving others under-covered.

Reference baselines (provided as `edit_ops`):
- **`confidence_thresholding`** — single global threshold on `max(softmax)` chosen so the overall acceptance rate matches the target coverage.
- **`conformal_abstention`** — split-conformal: compute non-conformity scores on calibration data and threshold so the achieved coverage is at least the target with high probability. Foundational reference: Vovk, Gammerman, Shafer, *Algorithmic Learning in a Random World*, 2005.
- **`learned_deferral`** — train a compact meta-classifier predicting whether the base model will be correct on a given example; defer those it predicts as "wrong". Related: Mozannar & Sontag, "Consistent Estimators for Learning to Defer", ICML 2020 ([arXiv:2006.01862](https://arxiv.org/abs/2006.01862)).
- **`groupwise_thresholding`** — per-subgroup thresholds, each tuned to hit the target coverage on its subgroup; a stronger fairness-aware reference.

## Implementation Contract
Implement `SelectivePolicy` in `scikit-learn/custom_selective.py`:

```python
class SelectivePolicy:
    def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT,
                 random_state: int = 0):
        ...

    def fit(self, probs: np.ndarray, y_true: np.ndarray,
            groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
        # probs: (n, n_classes) calibration-time base-model probabilities
        # y_true: (n,) calibration labels
        # groups: (n,) integer subgroup ids
        # X: optional raw features
        ...

    def acceptance_score(self, probs, groups, X=None) -> np.ndarray:
        # Higher score = more confident -> more likely to accept.
        ...

    def predict_accept(self, probs, groups, X=None) -> np.ndarray:
        # Boolean array: True = accept, False = defer.
        ...

    def calibration_summary(self) -> dict[str, float]:
        ...
```

You may implement a global threshold, a learned acceptance score, subgroup-specific thresholds, conformal mechanisms, or any compact policy fitting this interface. The base classifier and the train/calibration/test split are not editable.

## Fixed Pipeline
The data pipeline, base classifier, and the train / calibration / test split are fixed by the harness and not editable. The policy fits on the calibration outputs (probabilities/labels/subgroups) and is evaluated on the test split.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scikit-learn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `scikit-learn/custom_selective.py`
- editable lines **253–287**




## Readable Context


### `scikit-learn/custom_selective.py`  [EDITABLE — lines 253–287 only]

```python
     1: """Selective prediction / deferral benchmark.
     2: 
     3: Fixed:
     4: - offline AIF360 high-stakes dataset loading
     5: - train / calibration / test splits
     6: - base classifier training
     7: - metric computation
     8: 
     9: Editable:
    10: - SelectivePolicy, which decides whether to accept or defer predictions
    11:   based on calibration outputs.
    12: """
    13: 
    14: from __future__ import annotations
    15: 
    16: import argparse
    17: import json
    18: import os
    19: import warnings
    20: from dataclasses import dataclass
    21: from pathlib import Path
    22: from typing import Callable
    23: 
    24: import numpy as np
    25: import pandas as pd
    26: from sklearn.ensemble import GradientBoostingClassifier
    27: from sklearn.exceptions import ConvergenceWarning
    28: from sklearn.linear_model import LogisticRegression
    29: from sklearn.metrics import roc_auc_score
    30: from sklearn.model_selection import train_test_split
    31: from sklearn.pipeline import Pipeline
    32: from sklearn.preprocessing import StandardScaler
    33: 
    34: warnings.filterwarnings("ignore", category=ConvergenceWarning)
    35: 
    36: TARGET_COVERAGE_DEFAULT = 0.80
    37: DATA_HOME = os.environ.get("SKLEARN_DATA_HOME", "/data/sklearn")
    38: 
    39: 
    40: @dataclass(frozen=True)
    41: class BenchmarkSpec:
    42:     name: str
    43:     load_raw: Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray, bool]]
    44:     group_name: str
    45: 
    46: 
    47: def _dataset_parts(bundle):
    48:     if hasattr(bundle, "X") and hasattr(bundle, "y"):
    49:         return bundle.X, bundle.y
    50:     return bundle[0], bundle[1]
    51: 
    52: 
    53: def _as_frame(X) -> pd.DataFrame:
    54:     if isinstance(X, pd.DataFrame):
    55:         df = X.copy()
    56:         if any(name is not None for name in df.index.names):
    57:             idx_df = df.index.to_frame(index=False)
    58:             for col in reversed(list(idx_df.columns)):
    59:                 if col not in df.columns:
    60:                     df.insert(0, col, idx_df[col].to_numpy())
    61:         return df.reset_index(drop=True)
    62:     return pd.DataFrame(np.asarray(X)).reset_index(drop=True)
    63: 
    64: 
    65: def _encode_features(df: pd.DataFrame) -> np.ndarray:
    66:     encoded = pd.get_dummies(df, dummy_na=False)
    67:     encoded = encoded.replace([np.inf, -np.inf], np.nan)
    68:     encoded = encoded.fillna(encoded.median(numeric_only=True)).fillna(0.0)
    69:     return encoded.astype(np.float32).to_numpy()
    70: 
    71: 
    72: def _column_values(df: pd.DataFrame, candidates: list[str]) -> np.ndarray:
    73:     lower_map = {str(c).lower(): c for c in df.columns}
    74:     for name in candidates:
    75:         col = lower_map.get(str(name).lower())
    76:         if col is not None:
    77:             values = df[col]
    78:             numeric = pd.to_numeric(values, errors="coerce")
    79:             if numeric.notna().mean() >= 0.8:
    80:                 arr = numeric.to_numpy(dtype=float)
    81:                 med = np.nanmedian(arr)
    82:                 return np.nan_to_num(arr, nan=med)
    83:             return values.astype("category").cat.codes.to_numpy(dtype=float)
    84:     first = df.iloc[:, 0]
    85:     numeric = pd.to_numeric(first, errors="coerce")
    86:     if numeric.notna().mean() >= 0.8:
    87:         arr = numeric.to_numpy(dtype=float)
    88:         med = np.nanmedian(arr)
    89:         return np.nan_to_num(arr, nan=med)
    90:     return first.astype("category").cat.codes.to_numpy(dtype=float)
    91: 
    92: 
    93: def _protected_groups(df: pd.DataFrame, candidates: list[str]) -> np.ndarray:
    94:     lower_map = {str(c).lower(): c for c in df.columns}
    95:     code_columns = []
    96:     for name in candidates:
    97:         col = lower_map.get(str(name).lower())
    98:         if col is None:
    99:             continue
   100:         codes = pd.Series(df[col]).astype("category").cat.codes.to_numpy(dtype=int)
   101:         code_columns.append(codes)
   102:     if not code_columns:
   103:         return _quantile_bins(_column_values(df, [df.columns[0]]), n_bins=2)
   104: 
   105:     combined = np.zeros(len(df), dtype=int)
   106:     factor = 1
   107:     for codes in code_columns:
   108:         codes = codes - int(codes.min())
   109:         combined += factor * codes
   110:         factor *= int(codes.max()) + 1
   111:     unique = {value: idx for idx, value in enumerate(sorted(np.unique(combined)))}
   112:     return np.asarray([unique[value] for value in combined], dtype=int)
   113: 
   114: 
   115: def _binary_labels(y, positive_tokens: list[str] | None = None) -> tuple[np.ndarray, bool]:
   116:     series = pd.Series(y).reset_index(drop=True)
   117:     numeric = pd.to_numeric(series, errors="coerce")
   118:     if numeric.notna().mean() >= 0.95:
   119:         arr = numeric.to_numpy(dtype=float)
   120:         uniq = np.unique(arr[~np.isnan(arr)])
   121:         if len(uniq) <= 2:
   122:             return (arr == np.max(uniq)).astype(int), False
   123:         return arr.astype(np.float32), True
   124: 
   125:     text = series.astype(str).str.lower().str.strip()
   126:     if positive_tokens:
   127:         tokens = tuple(tok.lower() for tok in positive_tokens)
   128:         return text.apply(lambda value: any(tok in value for tok in tokens)).to_numpy(dtype=int), False
   129:     values = sorted(text.unique())
   130:     return (text == values[-1]).to_numpy(dtype=int), False
   131: 
   132: 
   133: def _load_adult() -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
   134:     from aif360.sklearn.datasets import fetch_adult
   135: 
   136:     X_raw, y_raw = _dataset_parts(
   137:         fetch_adult(data_home=DATA_HOME, cache=True, binary_race=True, dropna=True)
   138:     )
   139:     df = _as_frame(X_raw)
   140:     y, is_regression = _binary_labels(y_raw, positive_tokens=[">50k"])
   141:     groups = _protected_groups(df, ["sex", "race"])
   142:     return _encode_features(df), y, groups, is_regression
   143: 
   144: 
   145: def _load_compas() -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
   146:     from aif360.sklearn.datasets import fetch_compas
   147: 
   148:     X_raw, y_raw = _dataset_parts(
   149:         fetch_compas(data_home=DATA_HOME, cache=True, binary_race=True, dropna=True)
   150:     )
   151:     df = _as_frame(X_raw)
   152:     y, is_regression = _binary_labels(y_raw, positive_tokens=["survived", "no recid", "0"])
   153:     groups = _protected_groups(df, ["race", "sex"])
   154:     return _encode_features(df), y, groups, is_regression
   155: 
   156: 
   157: def _load_law_school() -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
   158:     from aif360.sklearn.datasets import fetch_lawschool_gpa
   159: 
   160:     X_raw, y_raw = _dataset_parts(
   161:         fetch_lawschool_gpa(data_home=DATA_HOME, cache=True, binary_race=True, dropna=True)
   162:     )
   163:     df = _as_frame(X_raw)
   164:     y, is_regression = _binary_labels(y_raw)
   165:     groups = _protected_groups(df, ["race", "gender"])
   166:     return _encode_features(df), y, groups, is_regression
   167: 
   168: 
   169: BENCHMARKS: dict[str, BenchmarkSpec] = {
   170:     "adult": BenchmarkSpec("adult", _load_adult, "sex/race"),
   171:     "compas": BenchmarkSpec("compas", _load_compas, "race/sex"),
   172:     "law_school": BenchmarkSpec("law_school", _load_law_school, "race/gender"),
   173: }
   174: 
   175: 
   176: def _safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
   177:     y_true = np.asarray(y_true, dtype=int)
   178:     if len(np.unique(y_true)) < 2:
   179:         return 0.5
   180:     if np.allclose(scores, scores[0]):
   181:         return 0.5
   182:     try:
   183:         return float(roc_auc_score(y_true, scores))
   184:     except ValueError:
   185:         return 0.5
   186: 
   187: 
   188: def _quantile_bins(values: np.ndarray, n_bins: int = 5) -> np.ndarray:
   189:     quantiles = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
   190:     quantiles = np.unique(quantiles)
   191:     if len(quantiles) <= 2:
   192:         return np.zeros(len(values), dtype=int)
   193:     return np.digitize(values, quantiles[1:-1], right=True)
   194: 
   195: 
   196: def _make_binary_targets(raw_y: np.ndarray, train_idx: np.ndarray, is_regression: bool) -> tuple[np.ndarray, float]:
   197:     if not is_regression:
   198:         return raw_y.astype(int), float("nan")
   199:     threshold = float(np.median(raw_y[train_idx]))
   200:     y = (raw_y > threshold).astype(int)
   201:     return y, threshold
   202: 
   203: 
   204: def _split_dataset(train_idx: np.ndarray, test_idx: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> dict[str, np.ndarray]:
   205:     n_groups = int(np.max(groups)) + 1
   206:     strata_train = y[train_idx] * n_groups + groups[train_idx]
   207:     counts = np.bincount(strata_train.astype(int))
   208:     stratify = strata_train if np.all(counts[counts > 0] >= 2) else None
   209:     fit_idx, cal_idx = train_test_split(train_idx, test_size=0.25, random_state=seed, stratify=stratify)
   210:     return {
   211:         "fit_idx": np.sort(fit_idx),
   212:         "cal_idx": np.sort(cal_idx),
   213:         "test_idx": np.sort(test_idx),
   214:     }
   215: 
   216: 
   217: def _build_base_model(seed: int) -> Pipeline:
   218:     return Pipeline(
   219:         steps=[
   220:             ("scale", StandardScaler()),
   221:             (
   222:                 "clf",
   223:                 GradientBoostingClassifier(
   224:                     n_estimators=200,
   225:                     max_depth=3,
   226:                     learning_rate=0.1,
   227:                     random_state=seed,
   228:                 ),
   229:             ),
   230:         ]
   231:     )
   232: 
   233: 
   234: def _confidence_features(probs: np.ndarray, groups: np.ndarray | None = None, X: np.ndarray | None = None) -> np.ndarray:
   235:     p1 = probs[:, 1]
   236:     p0 = probs[:, 0]
   237:     max_prob = np.maximum(p0, p1)
   238:     margin = np.abs(p1 - p0)
   239:     entropy = -(p0 * np.log(np.clip(p0, 1e-12, 1.0)) + p1 * np.log(np.clip(p1, 1e-12, 1.0)))
   240:     feats = [p1, max_prob, margin, entropy]
   241:     if groups is not None:
   242:         feats.append(groups.astype(float))
   243:     if X is not None and X.ndim == 2 and X.shape[1] > 0:
   244:         feats.append(X[:, 0].astype(float))
   245:     return np.column_stack(feats)
   246: 
   247: 
   248: # =============================================================================
   249: # EDITABLE REGION START
   250: # =============================================================================
   251: 
   252: 
   253: class SelectivePolicy:
   254:     """Policy that maps calibration outputs to accept / defer decisions.
   255: 
   256:     The default implementation is intentionally conservative:
   257:     it accepts the top-confidence examples needed to reach the target coverage.
   258:     Baselines replace this class with more specialized policies.
   259:     """
   260: 
   261:     def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT, random_state: int = 0):
   262:         self.target_coverage = float(target_coverage)
   263:         self.random_state = int(random_state)
   264:         self.threshold_: float = 0.5
   265:         self.group_thresholds_: dict[int, float] = {}
   266:         self.meta_model_ = None
   267:         self.strategy_name = "global_threshold"
   268: 
   269:     def fit(self, probs: np.ndarray, y_true: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
   270:         scores = self.acceptance_score(probs, groups, X)
   271:         quantile = float(np.clip(1.0 - self.target_coverage, 0.0, 1.0))
   272:         self.threshold_ = float(np.quantile(scores, quantile))
   273:         self.group_thresholds_ = {}
   274:         self.meta_model_ = None
   275:         return self
   276: 
   277:     def acceptance_score(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
   278:         return np.max(probs, axis=1)
   279: 
   280:     def predict_accept(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
   281:         scores = self.acceptance_score(probs, groups, X)
   282:         return scores >= self.threshold_
   283: 
   284:     def calibration_summary(self) -> dict[str, float]:
   285:         return {
   286:             "threshold": float(self.threshold_),
   287:         }
   288: 
   289: 
   290: # =============================================================================
   291: # EDITABLE REGION END
   292: # =============================================================================
   293: 
   294: 
   295: def _predict_labels(probs: np.ndarray) -> np.ndarray:
   296:     return probs.argmax(axis=1)
   297: 
   298: 
   299: def _selective_metrics(
   300:     y_true: np.ndarray,
   301:     y_pred: np.ndarray,
   302:     accept: np.ndarray,
   303:     scores: np.ndarray,
   304:     groups: np.ndarray,
   305: ) -> dict[str, float]:
   306:     accept = accept.astype(bool)
   307:     coverage = float(accept.mean())
   308:     if accept.any():
   309:         selective_risk = float(np.mean(y_pred[accept] != y_true[accept]))
   310:     else:
   311:         selective_risk = 1.0
   312: 
   313:     group_risks = []
   314:     group_deferrals = []
   315:     for group_id in np.unique(groups):
   316:         group_mask = groups == group_id
   317:         group_accept = accept[group_mask]
   318:         group_y = y_true[group_mask]
   319:         group_pred = y_pred[group_mask]
   320:         if group_mask.sum() == 0:
   321:             continue
   322:         if group_accept.any():
   323:             group_risk = float(np.mean(group_pred[group_accept] != group_y[group_accept]))
   324:         else:
   325:             group_risk = 1.0
   326:         group_risks.append(group_risk)
   327:         group_deferrals.append(float(1.0 - group_accept.mean()))
   328: 
   329:     worst_group_risk = float(max(group_risks)) if group_risks else selective_risk
   330:     deferral_gap = float(max(group_deferrals) - min(group_deferrals)) if group_deferrals else 0.0
   331:     correctness = (y_pred == y_true).astype(int)
   332:     auroc = _safe_roc_auc(correctness, scores)
   333:     return {
   334:         "selective_risk_at80": selective_risk,
   335:         "coverage_at80": coverage,
   336:         "worst_group_selective_risk": worst_group_risk,
   337:         "deferral_rate_gap": deferral_gap,
   338:         "auroc": auroc,
   339:     }
   340: 
   341: 
   342: def _print_metrics(prefix: str, metrics: dict[str, float]) -> None:
   343:     parts = [f"{key}={value:.6f}" for key, value in metrics.items()]
   344:     print(f"{prefix}: " + " ".join(parts), flush=True)
   345: 
   346: 
   347: def run_benchmark(dataset: str, seed: int, target_coverage: float, output_dir: str | None = None) -> dict[str, float]:
   348:     if dataset not in BENCHMARKS:
   349:         raise ValueError(f"Unknown dataset '{dataset}'. Expected one of: {sorted(BENCHMARKS)}")
   350: 
   351:     spec = BENCHMARKS[dataset]
   352:     X, raw_y, raw_groups, is_regression = spec.load_raw()
   353: 
   354:     indices = np.arange(len(X))
   355:     if is_regression:
   356:         stratify_for_split = _quantile_bins(raw_y, n_bins=5)
   357:     else:
   358:         stratify_for_split = raw_y.astype(int)
   359: 
   360:     train_idx, test_idx = train_test_split(
   361:         indices,
   362:         test_size=0.2,
   363:         random_state=seed,
   364:         stratify=stratify_for_split,
   365:     )
   366:     y, label_threshold = _make_binary_targets(raw_y, train_idx, is_regression=is_regression)
   367:     groups = np.asarray(raw_groups, dtype=int)
   368:     group_threshold = -1.0
   369: 
   370:     split = _split_dataset(train_idx, test_idx, y, groups, seed)
   371:     fit_idx = split["fit_idx"]
   372:     cal_idx = split["cal_idx"]
   373:     test_idx = split["test_idx"]
   374: 
   375:     model = _build_base_model(seed)
   376:     model.fit(X[fit_idx], y[fit_idx])
   377: 
   378:     cal_probs = model.predict_proba(X[cal_idx])
   379:     test_probs = model.predict_proba(X[test_idx])
   380:     cal_pred = _predict_labels(cal_probs)
   381:     test_pred = _predict_labels(test_probs)
   382: 
   383:     policy = SelectivePolicy(target_coverage=target_coverage, random_state=seed)
   384:     policy.fit(cal_probs, y[cal_idx], groups[cal_idx], X=X[cal_idx])
   385:     cal_accept = policy.predict_accept(cal_probs, groups[cal_idx], X=X[cal_idx])
   386:     test_accept = policy.predict_accept(test_probs, groups[test_idx], X=X[test_idx])
   387:     test_scores = policy.acceptance_score(test_probs, groups[test_idx], X=X[test_idx])
   388: 
   389:     train_acc = float(np.mean(model.predict(X[fit_idx]) == y[fit_idx]))
   390:     cal_acc = float(np.mean(cal_pred == y[cal_idx]))
   391:     train_summary = {
   392:         "train_accuracy": train_acc,
   393:         "cal_accuracy": cal_acc,
   394:         "cal_coverage": float(cal_accept.mean()),
   395:         "policy_threshold": float(getattr(policy, "threshold_", 0.0)),
   396:     }
   397:     _print_metrics("TRAIN_METRICS", train_summary)
   398: 
   399:     test_metrics = _selective_metrics(y[test_idx], test_pred, test_accept, test_scores, groups[test_idx])
   400:     test_metrics["target_coverage"] = float(target_coverage)
   401:     test_metrics["actual_coverage"] = float(test_accept.mean())
   402:     test_metrics["label_threshold"] = float(label_threshold) if np.isfinite(label_threshold) else -1.0
   403:     test_metrics["group_threshold"] = float(group_threshold)
   404:     _print_metrics("TEST_METRICS", test_metrics)
   405: 
   406:     if output_dir:
   407:         Path(output_dir).mkdir(parents=True, exist_ok=True)
   408:         summary_path = Path(output_dir) / f"{dataset}_summary.json"
   409:         with summary_path.open("w", encoding="utf-8") as f:
   410:             json.dump({"train": train_summary, "test": test_metrics}, f, indent=2, sort_keys=True)
   411: 
   412:     return test_metrics
   413: 
   414: 
   415: def main() -> None:
   416:     parser = argparse.ArgumentParser(description="Selective prediction / deferral benchmark.")
   417:     parser.add_argument(
   418:         "--dataset",
   419:         required=True,
   420:         choices=sorted(BENCHMARKS),
   421:         help="Benchmark dataset name.",
   422:     )
   423:     parser.add_argument("--seed", type=int, default=42)
   424:     parser.add_argument("--target-coverage", type=float, default=TARGET_COVERAGE_DEFAULT)
   425:     parser.add_argument("--output-dir", type=str, default=None)
   426:     args = parser.parse_args()
   427: 
   428:     run_benchmark(args.dataset, args.seed, args.target_coverage, args.output_dir)
   429: 
   430: 
   431: if __name__ == "__main__":
   432:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `confidence_thresholding` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_selective.py`:

```python
Lines 253–279:
   250: # =============================================================================
   251: 
   252: 
   253: class SelectivePolicy:
   254:     """Global confidence threshold tuned on the calibration set."""
   255: 
   256:     def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT, random_state: int = 0):
   257:         self.target_coverage = float(target_coverage)
   258:         self.random_state = int(random_state)
   259:         self.threshold_: float = 0.5
   260:         self.group_thresholds_: dict[int, float] = {}
   261:         self.meta_model_ = None
   262:         self.strategy_name = "confidence_thresholding"
   263: 
   264:     def fit(self, probs: np.ndarray, y_true: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
   265:         scores = self.acceptance_score(probs, groups, X)
   266:         quantile = float(np.clip(1.0 - self.target_coverage, 0.0, 1.0))
   267:         self.threshold_ = float(np.quantile(scores, quantile))
   268:         self.group_thresholds_ = {}
   269:         self.meta_model_ = None
   270:         return self
   271: 
   272:     def acceptance_score(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
   273:         return np.max(probs, axis=1)
   274: 
   275:     def predict_accept(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
   276:         return self.acceptance_score(probs, groups, X) >= self.threshold_
   277: 
   278:     def calibration_summary(self) -> dict[str, float]:
   279:         return {"threshold": float(self.threshold_)}
   280: 
   281: 
   282: # =============================================================================
```

### `conformal_abstention` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_selective.py`:

```python
Lines 253–283:
   250: # =============================================================================
   251: 
   252: 
   253: class SelectivePolicy:
   254:     """Conformal abstention using a held-out calibration set."""
   255: 
   256:     def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT, random_state: int = 0):
   257:         self.target_coverage = float(target_coverage)
   258:         self.random_state = int(random_state)
   259:         self.threshold_: float = 0.5
   260:         self.group_thresholds_: dict[int, float] = {}
   261:         self.meta_model_ = None
   262:         self.strategy_name = "conformal_abstention"
   263: 
   264:     def fit(self, probs: np.ndarray, y_true: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
   265:         scores = np.max(probs, axis=1)
   266:         nonconformity = 1.0 - scores
   267:         n = len(nonconformity)
   268:         alpha = float(np.clip(1.0 - self.target_coverage, 0.0, 1.0))
   269:         rank = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
   270:         rank = int(np.clip(rank, 0, n - 1))
   271:         self.threshold_ = float(1.0 - np.sort(nonconformity)[rank])
   272:         self.group_thresholds_ = {}
   273:         self.meta_model_ = None
   274:         return self
   275: 
   276:     def acceptance_score(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
   277:         return np.max(probs, axis=1)
   278: 
   279:     def predict_accept(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
   280:         return self.acceptance_score(probs, groups, X) >= self.threshold_
   281: 
   282:     def calibration_summary(self) -> dict[str, float]:
   283:         return {"threshold": float(self.threshold_)}
   284: 
   285: 
   286: # =============================================================================
```

### `learned_deferral` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_selective.py`:

```python
Lines 253–298:
   250: # =============================================================================
   251: 
   252: 
   253: class SelectivePolicy:
   254:     """Compact learned gate that predicts correctness from confidence features."""
   255: 
   256:     def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT, random_state: int = 0):
   257:         self.target_coverage = float(target_coverage)
   258:         self.random_state = int(random_state)
   259:         self.threshold_: float = 0.5
   260:         self.group_thresholds_: dict[int, float] = {}
   261:         self.meta_model_ = None
   262:         self.strategy_name = "learned_deferral"
   263: 
   264:     def fit(self, probs: np.ndarray, y_true: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
   265:         features = _confidence_features(probs, groups, X)
   266:         correct = (np.argmax(probs, axis=1) == y_true).astype(int)
   267:         self.meta_model_ = Pipeline(
   268:             steps=[
   269:                 ("scale", StandardScaler()),
   270:                 (
   271:                     "clf",
   272:                     LogisticRegression(
   273:                         max_iter=1000,
   274:                         solver="lbfgs",
   275:                         class_weight="balanced",
   276:                         random_state=self.random_state,
   277:                     ),
   278:                 ),
   279:             ]
   280:         )
   281:         self.meta_model_.fit(features, correct)
   282:         scores = self.meta_model_.predict_proba(features)[:, 1]
   283:         quantile = float(np.clip(1.0 - self.target_coverage, 0.0, 1.0))
   284:         self.threshold_ = float(np.quantile(scores, quantile))
   285:         self.group_thresholds_ = {}
   286:         return self
   287: 
   288:     def acceptance_score(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
   289:         if self.meta_model_ is None:
   290:             return np.max(probs, axis=1)
   291:         features = _confidence_features(probs, groups, X)
   292:         return self.meta_model_.predict_proba(features)[:, 1]
   293: 
   294:     def predict_accept(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
   295:         return self.acceptance_score(probs, groups, X) >= self.threshold_
   296: 
   297:     def calibration_summary(self) -> dict[str, float]:
   298:         return {"threshold": float(self.threshold_)}
   299: 
   300: 
   301: # =============================================================================
```

### `groupwise_thresholding` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_selective.py`:

```python
Lines 253–289:
   250: # =============================================================================
   251: 
   252: 
   253: class SelectivePolicy:
   254:     """Subgroup-specific thresholds tuned on calibration data."""
   255: 
   256:     def __init__(self, target_coverage: float = TARGET_COVERAGE_DEFAULT, random_state: int = 0):
   257:         self.target_coverage = float(target_coverage)
   258:         self.random_state = int(random_state)
   259:         self.threshold_: float = 0.5
   260:         self.group_thresholds_: dict[int, float] = {}
   261:         self.meta_model_ = None
   262:         self.strategy_name = "groupwise_thresholding"
   263: 
   264:     def fit(self, probs: np.ndarray, y_true: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> "SelectivePolicy":
   265:         scores = self.acceptance_score(probs, groups, X)
   266:         quantile = float(np.clip(1.0 - self.target_coverage, 0.0, 1.0))
   267:         self.threshold_ = float(np.quantile(scores, quantile))
   268:         self.group_thresholds_ = {}
   269:         for group_id in np.unique(groups):
   270:             mask = groups == group_id
   271:             if not np.any(mask):
   272:                 continue
   273:             self.group_thresholds_[int(group_id)] = float(np.quantile(scores[mask], quantile))
   274:         self.meta_model_ = None
   275:         return self
   276: 
   277:     def acceptance_score(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
   278:         return np.max(probs, axis=1)
   279: 
   280:     def predict_accept(self, probs: np.ndarray, groups: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
   281:         scores = self.acceptance_score(probs, groups, X)
   282:         thresholds = np.asarray([self.group_thresholds_.get(int(group), self.threshold_) for group in groups], dtype=float)
   283:         return scores >= thresholds
   284: 
   285:     def calibration_summary(self) -> dict[str, float]:
   286:         summary = {"threshold": float(self.threshold_)}
   287:         for group_id, threshold in self.group_thresholds_.items():
   288:             summary[f"threshold_group_{group_id}"] = float(threshold)
   289:         return summary
   290: 
   291: 
   292: # =============================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
