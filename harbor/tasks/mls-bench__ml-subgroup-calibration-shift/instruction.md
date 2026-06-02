# MLS-Bench: ml-subgroup-calibration-shift

# Subgroup Calibration Under Distribution Shift

## Research Question
Design a post-hoc calibration method that remains reliable across subgroups when the test distribution shifts relative to calibration. The base tabular classifier and the (intentionally shifted) train/calibration/test splits are fixed; the contribution is the *calibration mapping* applied to positive-class probabilities, optionally using subgroup IDs.

## Background
Many calibration methods look accurate on average while remaining unreliable for protected or operationally meaningful subgroups, especially once subgroup prevalence or score distribution shifts at test time. The challenge is to improve worst-subgroup calibration without overfitting small per-group calibration samples.

Reference baselines:
- **Temperature scaling** — Guo, Pleiss, Sun, Weinberger, ICML 2017 ([arXiv:1706.04599](https://arxiv.org/abs/1706.04599)). Single global scalar `T` divides logits before sigmoid; fit by NLL on the calibration set.
- **Isotonic regression** — Zadrozny & Elkan, KDD 2002. Non-parametric monotonic mapping of probabilities to empirical accuracies.
- **Beta calibration** — Kull, Silva Filho, Flach, AISTATS 2017 ([proceedings](https://proceedings.mlr.press/v54/kull17a.html)). Three-parameter beta-distribution mapping; subsumes sigmoids, inverse sigmoids, and identity.
- **Group-wise temperature scaling** — fit one temperature per subgroup (with optional shrinkage toward the global temperature for small groups).

## Implementation Contract
Modify `CalibrationMethod` in `scikit-learn/custom_subgroup_calibration.py`:

```python
class CalibrationMethod:
    def fit(self, probs, labels, groups=None):
        # probs: (n,) positive-class probabilities from the base classifier
        # labels: (n,) integer labels {0,1}
        # groups: (n,) integer subgroup IDs (may be None for group-agnostic methods)
        return self

    def predict_proba(self, probs, groups=None):
        # Returns (n,) calibrated positive-class probabilities in [0, 1].
        ...
```

The method must produce valid probabilities; `groups` may be ignored by group-agnostic methods.

## Fixed Pipeline & Evaluation
Datasets (cached high-stakes tabular data from AIF360):
- **Adult** — Census income; subgroup attributes: sex, race.
- **COMPAS** — ProPublica recidivism risk; subgroup attributes: race, sex.
- **Law School GPA** — admissions/outcome data binarized at the median first-year GPA; subgroup attributes: race, gender.

For each dataset the test split is intentionally shifted: a domain score selects the held-out test tail, and calibration is fit on the source region and evaluated on the shifted region. Subgroups come from protected attributes exposed by the dataset loaders.

Metrics:
- **`worst_group_ece`** — worst-subgroup expected calibration error (lower is better).
- **`brier`** — Brier score on test (lower is better).
- **`max_subgroup_gap`** — max over subgroups of `|accuracy − mean confidence|` (lower is better).
- **`subgroup_auroc`** — subgroup-level AUROC (higher is better; reported diagnostically).


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scikit-learn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `scikit-learn/custom_subgroup_calibration.py`
- editable lines **200–219**




## Readable Context


### `scikit-learn/custom_subgroup_calibration.py`  [EDITABLE — lines 200–219 only]

```python
     1: """Subgroup calibration under distribution shift.
     2: 
     3: The benchmark uses cached high-stakes tabular datasets from AIF360 rather than
     4: task-local sklearn proxies. Package-level data preparation downloads the needed
     5: Adult, COMPAS, and Law School assets before compute-node execution.
     6: 
     7: Fixed:
     8: - dataset loading
     9: - shifted train/calibration/test split
    10: - base classifier training
    11: - metric computation
    12: 
    13: Editable:
    14: - CalibrationMethod
    15: """
    16: 
    17: import argparse
    18: import os
    19: import warnings
    20: 
    21: import numpy as np
    22: import pandas as pd
    23: from scipy import optimize, special
    24: from sklearn.isotonic import IsotonicRegression
    25: from sklearn.linear_model import LogisticRegression
    26: from sklearn.metrics import brier_score_loss, roc_auc_score
    27: from sklearn.pipeline import Pipeline
    28: from sklearn.preprocessing import StandardScaler
    29: 
    30: warnings.filterwarnings("ignore")
    31: 
    32: DATA_HOME = os.environ.get("SKLEARN_DATA_HOME", "/data/sklearn")
    33: 
    34: 
    35: def expected_calibration_error(probs, labels, n_bins=15):
    36:     probs = np.asarray(probs).reshape(-1)
    37:     labels = np.asarray(labels).reshape(-1).astype(int)
    38:     confidences = np.where(probs >= 0.5, probs, 1.0 - probs)
    39:     predictions = (probs >= 0.5).astype(int)
    40:     accuracies = (predictions == labels).astype(float)
    41: 
    42:     bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    43:     ece = 0.0
    44:     for i in range(n_bins):
    45:         lo, hi = bin_edges[i], bin_edges[i + 1]
    46:         mask = (confidences > lo) & (confidences <= hi)
    47:         prop = float(mask.mean())
    48:         if prop > 0:
    49:             ece += prop * abs(float(accuracies[mask].mean()) - float(confidences[mask].mean()))
    50:     return float(ece)
    51: 
    52: 
    53: def _safe_auc(labels, probs):
    54:     labels = np.asarray(labels).reshape(-1).astype(int)
    55:     probs = np.asarray(probs).reshape(-1)
    56:     if np.unique(labels).size < 2:
    57:         return float("nan")
    58:     return float(roc_auc_score(labels, probs))
    59: 
    60: 
    61: def _quantile_groups(score, n_groups=4):
    62:     score = np.asarray(score).reshape(-1)
    63:     edges = np.quantile(score, np.linspace(0.0, 1.0, n_groups + 1))
    64:     edges[0] = -np.inf
    65:     edges[-1] = np.inf
    66:     return np.digitize(score, edges[1:-1], right=True).astype(int)
    67: 
    68: 
    69: def _binary_threshold(target):
    70:     target = np.asarray(target).reshape(-1)
    71:     return (target > np.median(target)).astype(int)
    72: 
    73: 
    74: def _dataset_parts(bundle):
    75:     if hasattr(bundle, "X") and hasattr(bundle, "y"):
    76:         return bundle.X, bundle.y
    77:     return bundle[0], bundle[1]
    78: 
    79: 
    80: def _as_frame(X):
    81:     if isinstance(X, pd.DataFrame):
    82:         df = X.copy()
    83:         if any(name is not None for name in df.index.names):
    84:             idx_df = df.index.to_frame(index=False)
    85:             for col in reversed(list(idx_df.columns)):
    86:                 if col not in df.columns:
    87:                     df.insert(0, col, idx_df[col].to_numpy())
    88:         return df.reset_index(drop=True)
    89:     return pd.DataFrame(np.asarray(X)).reset_index(drop=True)
    90: 
    91: 
    92: def _encode_features(df):
    93:     encoded = pd.get_dummies(df, dummy_na=False)
    94:     encoded = encoded.replace([np.inf, -np.inf], np.nan)
    95:     encoded = encoded.fillna(encoded.median(numeric_only=True)).fillna(0.0)
    96:     return encoded.astype(np.float32).to_numpy()
    97: 
    98: 
    99: def _column_values(df, candidates):
   100:     lower_map = {str(c).lower(): c for c in df.columns}
   101:     for name in candidates:
   102:         col = lower_map.get(str(name).lower())
   103:         if col is not None:
   104:             values = df[col]
   105:             numeric = pd.to_numeric(values, errors="coerce")
   106:             if numeric.notna().mean() >= 0.8:
   107:                 arr = numeric.to_numpy(dtype=float)
   108:                 med = np.nanmedian(arr)
   109:                 return np.nan_to_num(arr, nan=med)
   110:             return values.astype("category").cat.codes.to_numpy(dtype=float)
   111:     first = df.iloc[:, 0]
   112:     numeric = pd.to_numeric(first, errors="coerce")
   113:     if numeric.notna().mean() >= 0.8:
   114:         arr = numeric.to_numpy(dtype=float)
   115:         med = np.nanmedian(arr)
   116:         return np.nan_to_num(arr, nan=med)
   117:     return first.astype("category").cat.codes.to_numpy(dtype=float)
   118: 
   119: 
   120: def _protected_groups(df, candidates):
   121:     lower_map = {str(c).lower(): c for c in df.columns}
   122:     code_columns = []
   123:     for name in candidates:
   124:         col = lower_map.get(str(name).lower())
   125:         if col is None:
   126:             continue
   127:         codes = pd.Series(df[col]).astype("category").cat.codes.to_numpy(dtype=int)
   128:         code_columns.append(codes)
   129:     if not code_columns:
   130:         return _quantile_groups(_column_values(df, [df.columns[0]]), n_groups=2)
   131: 
   132:     combined = np.zeros(len(df), dtype=int)
   133:     factor = 1
   134:     for codes in code_columns:
   135:         codes = codes - int(codes.min())
   136:         combined += factor * codes
   137:         factor *= int(codes.max()) + 1
   138:     unique = {value: idx for idx, value in enumerate(sorted(np.unique(combined)))}
   139:     return np.asarray([unique[value] for value in combined], dtype=int)
   140: 
   141: 
   142: def _binary_labels(y, positive_tokens=None):
   143:     series = pd.Series(y).reset_index(drop=True)
   144:     numeric = pd.to_numeric(series, errors="coerce")
   145:     if numeric.notna().mean() >= 0.95:
   146:         arr = numeric.to_numpy(dtype=float)
   147:         uniq = np.unique(arr[~np.isnan(arr)])
   148:         if len(uniq) <= 2:
   149:             return (arr == np.max(uniq)).astype(int)
   150:         return _binary_threshold(arr)
   151: 
   152:     text = series.astype(str).str.lower().str.strip()
   153:     if positive_tokens:
   154:         tokens = tuple(tok.lower() for tok in positive_tokens)
   155:         return text.apply(lambda value: any(tok in value for tok in tokens)).to_numpy(dtype=int)
   156:     values = sorted(text.unique())
   157:     positive = values[-1]
   158:     return (text == positive).to_numpy(dtype=int)
   159: 
   160: 
   161: def _load_adult():
   162:     from aif360.sklearn.datasets import fetch_adult
   163: 
   164:     X_raw, y_raw = _dataset_parts(
   165:         fetch_adult(data_home=DATA_HOME, cache=True, binary_race=True, dropna=True)
   166:     )
   167:     df = _as_frame(X_raw)
   168:     y = _binary_labels(y_raw, positive_tokens=[">50k"])
   169:     domain_score = _column_values(df, ["age", "education-num", "hours-per-week"])
   170:     groups = _protected_groups(df, ["sex", "race"])
   171:     return _encode_features(df), y, domain_score, groups
   172: 
   173: 
   174: def _load_compas():
   175:     from aif360.sklearn.datasets import fetch_compas
   176: 
   177:     X_raw, y_raw = _dataset_parts(
   178:         fetch_compas(data_home=DATA_HOME, cache=True, binary_race=True, dropna=True)
   179:     )
   180:     df = _as_frame(X_raw)
   181:     y = _binary_labels(y_raw, positive_tokens=["survived", "no recid", "0"])
   182:     domain_score = _column_values(df, ["priors_count", "age"])
   183:     groups = _protected_groups(df, ["race", "sex"])
   184:     return _encode_features(df), y, domain_score, groups
   185: 
   186: 
   187: def _load_law_school():
   188:     from aif360.sklearn.datasets import fetch_lawschool_gpa
   189: 
   190:     X_raw, y_raw = _dataset_parts(
   191:         fetch_lawschool_gpa(data_home=DATA_HOME, cache=True, binary_race=True, dropna=True)
   192:     )
   193:     df = _as_frame(X_raw)
   194:     y = _binary_labels(y_raw)
   195:     domain_score = _column_values(df, ["lsat", "ugpa"])
   196:     groups = _protected_groups(df, ["race", "gender"])
   197:     return _encode_features(df), y, domain_score, groups
   198: 
   199: 
   200: class CalibrationMethod:
   201:     """Editable calibration method.
   202: 
   203:     Implement fit() and predict_proba() to map raw positive-class probabilities
   204:     to calibrated positive-class probabilities.
   205:     """
   206: 
   207:     def __init__(self):
   208:         self.eps = 1e-6
   209:         self._identity = True
   210: 
   211:     def fit(self, probs, labels, groups=None):
   212:         probs = np.asarray(probs).reshape(-1)
   213:         labels = np.asarray(labels).reshape(-1).astype(int)
   214:         self._base_rate = float(np.clip(labels.mean(), self.eps, 1.0 - self.eps))
   215:         return self
   216: 
   217:     def predict_proba(self, probs, groups=None):
   218:         probs = np.asarray(probs).reshape(-1)
   219:         return np.clip(probs, self.eps, 1.0 - self.eps)
   220: 
   221: 
   222: def _load_dataset(name):
   223:     loaders = {
   224:         "adult": _load_adult,
   225:         "compas": _load_compas,
   226:         "law_school": _load_law_school,
   227:     }
   228:     if name not in loaders:
   229:         raise ValueError(f"Unknown dataset: {name}")
   230:     return loaders[name]()
   231: 
   232: 
   233: def _shifted_split(y, domain_score, seed, test_frac=0.30, calib_frac=0.25):
   234:     rng = np.random.RandomState(seed)
   235:     train_idx = []
   236:     calib_idx = []
   237:     test_idx = []
   238: 
   239:     for cls in np.unique(y):
   240:         cls_idx = np.flatnonzero(y == cls)
   241:         order = cls_idx[np.argsort(domain_score[cls_idx])]
   242:         n_test = max(1, int(round(test_frac * len(order))))
   243:         source_idx = order[:-n_test]
   244:         test_cls = order[-n_test:]
   245:         n_cal = max(1, int(round(calib_frac * len(source_idx))))
   246:         calib_cls = rng.choice(source_idx, size=n_cal, replace=False)
   247:         train_cls = np.setdiff1d(source_idx, calib_cls, assume_unique=False)
   248: 
   249:         train_idx.append(train_cls)
   250:         calib_idx.append(calib_cls)
   251:         test_idx.append(test_cls)
   252: 
   253:     train_idx = np.sort(np.concatenate(train_idx))
   254:     calib_idx = np.sort(np.concatenate(calib_idx))
   255:     test_idx = np.sort(np.concatenate(test_idx))
   256:     return train_idx, calib_idx, test_idx
   257: 
   258: 
   259: def _fit_base_classifier(X_train, y_train, seed):
   260:     model = Pipeline(
   261:         steps=[
   262:             ("scale", StandardScaler()),
   263:             (
   264:                 "clf",
   265:                 LogisticRegression(
   266:                     max_iter=1200,
   267:                     solver="lbfgs",
   268:                     class_weight="balanced",
   269:                     random_state=seed,
   270:                 ),
   271:             ),
   272:         ]
   273:     )
   274:     model.fit(X_train, y_train)
   275:     return model
   276: 
   277: 
   278: def _evaluate(probs, labels, groups):
   279:     probs = np.asarray(probs).reshape(-1)
   280:     labels = np.asarray(labels).reshape(-1).astype(int)
   281:     groups = np.asarray(groups).reshape(-1).astype(int)
   282: 
   283:     group_ece = []
   284:     group_auc = []
   285:     for g in np.unique(groups):
   286:         mask = groups == g
   287:         if mask.sum() < 5:
   288:             continue
   289:         group_ece.append(expected_calibration_error(probs[mask], labels[mask]))
   290:         group_auc.append(_safe_auc(labels[mask], probs[mask]))
   291: 
   292:     worst_group_ece = float(np.max(group_ece)) if group_ece else float("nan")
   293:     subgroup_auroc = float(np.nanmean(group_auc)) if group_auc else float("nan")
   294:     max_subgroup_gap = float(np.max(group_ece) - np.min(group_ece)) if len(group_ece) > 1 else float("nan")
   295:     brier = float(brier_score_loss(labels, probs))
   296:     return {
   297:         "worst_group_ece": worst_group_ece,
   298:         "brier": brier,
   299:         "subgroup_auroc": subgroup_auroc,
   300:         "max_subgroup_gap": max_subgroup_gap,
   301:     }
   302: 
   303: 
   304: def main():
   305:     parser = argparse.ArgumentParser()
   306:     parser.add_argument("--dataset", choices=["adult", "compas", "law_school"], required=True)
   307:     parser.add_argument("--seed", type=int, default=42)
   308:     parser.add_argument("--output-dir", default="./output")
   309:     args = parser.parse_args()
   310: 
   311:     os.makedirs(args.output_dir, exist_ok=True)
   312: 
   313:     X, y, domain_score, groups = _load_dataset(args.dataset)
   314:     train_idx, calib_idx, test_idx = _shifted_split(y, domain_score, seed=args.seed)
   315: 
   316:     model = _fit_base_classifier(X[train_idx], y[train_idx], seed=args.seed)
   317:     cal_probs = model.predict_proba(X[calib_idx])[:, 1]
   318:     test_probs = model.predict_proba(X[test_idx])[:, 1]
   319: 
   320:     method = CalibrationMethod().fit(cal_probs, y[calib_idx], groups=groups[calib_idx])
   321:     cal_probs_hat = method.predict_proba(cal_probs, groups=groups[calib_idx])
   322:     test_probs_hat = method.predict_proba(test_probs, groups=groups[test_idx])
   323: 
   324:     print(
   325:         "TRAIN_METRICS: "
   326:         f"dataset={args.dataset} "
   327:         f"cal_ece_before={expected_calibration_error(cal_probs, y[calib_idx]):.6f} "
   328:         f"cal_ece_after={expected_calibration_error(cal_probs_hat, y[calib_idx]):.6f} "
   329:         f"cal_brier_before={brier_score_loss(y[calib_idx], cal_probs):.6f} "
   330:         f"cal_brier_after={brier_score_loss(y[calib_idx], cal_probs_hat):.6f}",
   331:         flush=True,
   332:     )
   333: 
   334:     test_metrics = _evaluate(test_probs_hat, y[test_idx], groups[test_idx])
   335:     print(
   336:         "TEST_METRICS: "
   337:         + " ".join(f"{k}={v:.6f}" for k, v in test_metrics.items()),
   338:         flush=True,
   339:     )
   340: 
   341: 
   342: if __name__ == "__main__":
   343:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `temperature_scaling` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_subgroup_calibration.py`:

```python
Lines 200–225:
   197:     return _encode_features(df), y, domain_score, groups
   198: 
   199: 
   200: class CalibrationMethod:
   201:     """Global temperature scaling on positive-class probabilities."""
   202: 
   203:     def __init__(self):
   204:         self.eps = 1e-6
   205:         self.temperature_ = 1.0
   206: 
   207:     def fit(self, probs, labels, groups=None):
   208:         probs = np.asarray(probs).reshape(-1)
   209:         labels = np.asarray(labels).reshape(-1).astype(int)
   210:         logits = special.logit(np.clip(probs, self.eps, 1.0 - self.eps))
   211: 
   212:         def objective(log_t):
   213:             t = float(np.exp(log_t))
   214:             cal = special.expit(logits / t)
   215:             p = np.clip(cal, self.eps, 1.0 - self.eps)
   216:             return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))
   217: 
   218:         result = optimize.minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
   219:         self.temperature_ = float(np.exp(result.x)) if result.success else 1.0
   220:         return self
   221: 
   222:     def predict_proba(self, probs, groups=None):
   223:         probs = np.asarray(probs).reshape(-1)
   224:         logits = special.logit(np.clip(probs, self.eps, 1.0 - self.eps))
   225:         return np.clip(special.expit(logits / self.temperature_), self.eps, 1.0 - self.eps)
   226: 
   227: 
   228: def _load_dataset(name):
```

### `isotonic_regression` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_subgroup_calibration.py`:

```python
Lines 200–215:
   197:     return _encode_features(df), y, domain_score, groups
   198: 
   199: 
   200: class CalibrationMethod:
   201:     """Isotonic regression calibration."""
   202: 
   203:     def __init__(self):
   204:         self.eps = 1e-6
   205:         self.model_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
   206: 
   207:     def fit(self, probs, labels, groups=None):
   208:         probs = np.asarray(probs).reshape(-1)
   209:         labels = np.asarray(labels).reshape(-1).astype(int)
   210:         self.model_.fit(probs, labels)
   211:         return self
   212: 
   213:     def predict_proba(self, probs, groups=None):
   214:         probs = np.asarray(probs).reshape(-1)
   215:         return np.clip(self.model_.predict(probs), self.eps, 1.0 - self.eps)
   216: 
   217: 
   218: def _load_dataset(name):
```

### `beta_calibration` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_subgroup_calibration.py`:

```python
Lines 200–220:
   197:     return _encode_features(df), y, domain_score, groups
   198: 
   199: 
   200: class CalibrationMethod:
   201:     """Beta calibration via logistic regression on transformed probabilities."""
   202: 
   203:     def __init__(self):
   204:         self.eps = 1e-6
   205:         self.model_ = LogisticRegression(max_iter=2000, solver="lbfgs")
   206: 
   207:     def _featurize(self, probs):
   208:         probs = np.asarray(probs).reshape(-1)
   209:         p = np.clip(probs, self.eps, 1.0 - self.eps)
   210:         return np.column_stack([np.log(p), np.log1p(-p)])
   211: 
   212:     def fit(self, probs, labels, groups=None):
   213:         X = self._featurize(probs)
   214:         labels = np.asarray(labels).reshape(-1).astype(int)
   215:         self.model_.fit(X, labels)
   216:         return self
   217: 
   218:     def predict_proba(self, probs, groups=None):
   219:         X = self._featurize(probs)
   220:         return np.clip(self.model_.predict_proba(X)[:, 1], self.eps, 1.0 - self.eps)
   221: 
   222: 
   223: def _load_dataset(name):
```

### `group_temperature_scaling` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_subgroup_calibration.py`:

```python
Lines 200–257:
   197:     return _encode_features(df), y, domain_score, groups
   198: 
   199: 
   200: class CalibrationMethod:
   201:     """Group temperature scaling with James-Stein shrinkage to global T."""
   202: 
   203:     def __init__(self):
   204:         self.eps = 1e-6
   205:         self.k_shrink = 200.0
   206:         self.group_temperatures_ = {}
   207:         self.global_temperature_ = 1.0
   208: 
   209:     def _fit_temperature(self, probs, labels):
   210:         probs = np.asarray(probs).reshape(-1)
   211:         labels = np.asarray(labels).reshape(-1).astype(int)
   212:         logits = special.logit(np.clip(probs, self.eps, 1.0 - self.eps))
   213: 
   214:         def objective(log_t):
   215:             t = float(np.exp(log_t))
   216:             cal = special.expit(logits / t)
   217:             p = np.clip(cal, self.eps, 1.0 - self.eps)
   218:             return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))
   219: 
   220:         result = optimize.minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
   221:         return float(np.exp(result.x)) if result.success else 1.0
   222: 
   223:     def fit(self, probs, labels, groups=None):
   224:         probs = np.asarray(probs).reshape(-1)
   225:         labels = np.asarray(labels).reshape(-1).astype(int)
   226:         self.global_temperature_ = self._fit_temperature(probs, labels)
   227:         log_T_global = float(np.log(self.global_temperature_))
   228:         self.group_temperatures_ = {}
   229:         if groups is None:
   230:             return self
   231:         groups = np.asarray(groups).reshape(-1)
   232:         for g in np.unique(groups):
   233:             mask = groups == g
   234:             n_g = int(mask.sum())
   235:             if n_g < 20 or np.unique(labels[mask]).size < 2:
   236:                 self.group_temperatures_[int(g)] = self.global_temperature_
   237:                 continue
   238:             T_local = self._fit_temperature(probs[mask], labels[mask])
   239:             log_T_local = float(np.log(T_local))
   240:             alpha = n_g / (n_g + self.k_shrink)
   241:             log_T_g = alpha * log_T_local + (1.0 - alpha) * log_T_global
   242:             self.group_temperatures_[int(g)] = float(np.exp(log_T_g))
   243:         return self
   244: 
   245:     def predict_proba(self, probs, groups=None):
   246:         probs = np.asarray(probs).reshape(-1)
   247:         logits = special.logit(np.clip(probs, self.eps, 1.0 - self.eps))
   248:         if groups is None:
   249:             temp = self.global_temperature_
   250:             return np.clip(special.expit(logits / temp), self.eps, 1.0 - self.eps)
   251:         groups = np.asarray(groups).reshape(-1)
   252:         out = np.empty_like(probs)
   253:         for g in np.unique(groups):
   254:             mask = groups == g
   255:             temp = self.group_temperatures_.get(int(g), self.global_temperature_)
   256:             out[mask] = special.expit(logits[mask] / temp)
   257:         return np.clip(out, self.eps, 1.0 - self.eps)
   258: 
   259: 
   260: def _load_dataset(name):
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
