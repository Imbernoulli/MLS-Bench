# MLS-Bench: ml-missing-data-imputation

# Missing Data Imputation

## Research Question
Design a tabular missing-data imputation method that achieves low reconstruction error and preserves downstream predictive performance across diverse datasets. The contribution is the *imputer itself*: how feature dependencies are exploited, how imputations are iterated/refined, and how completed values are produced from data containing NaNs.

## Background
Missing data is ubiquitous. Mean/median imputation ignores feature correlations; iterative predictive methods exploit them.

Reference baselines:
- **Mean imputation** — replace NaNs in each column with the column mean from training data.
- **k-Nearest Neighbors imputation** — Troyanskaya et al., Bioinformatics 2001. For each missing entry, average over the `k` most similar rows (computed on observed features). Default `n_neighbors=5`.
- **MICE (Multivariate Imputation by Chained Equations)** — van Buuren & Groothuis-Oudshoorn, JSS 2011 ([paper](https://www.jstatsoft.org/v45/i03/)). Iterative: at each round and for each variable with missingness, fit a regression of that variable on all others (using the latest imputations) and replace its missing values with predictions. `sklearn.impute.IterativeImputer` is the de-facto MICE implementation; default `max_iter=10`.
- **MissForest** — Stekhoven & Bühlmann, Bioinformatics 2012 ([paper](https://academic.oup.com/bioinformatics/article/28/1/112/219101)). Iterative random-forest-based imputation; same chained-equations skeleton as MICE but uses a Random Forest as the per-variable predictor. Handles mixed-type data and complex interactions.
- **GAIN (Generative Adversarial Imputation Nets)** — Yoon, Jordon, van der Schaar, ICML 2018 ([arXiv:1806.02920](https://arxiv.org/abs/1806.02920)). GAN-based: generator imputes missing entries conditional on observed ones; discriminator tries to identify which entries were imputed; a hint mechanism reveals partial mask information.

## Implementation Contract
Implement `CustomImputer` in `scikit-learn/custom_imputation.py`:

```python
class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=42, max_iter=10):
        ...

    def fit(self, X, y=None):
        # X: numpy array (n_samples, n_features) with NaN for missing values.
        # Learn imputation model. Must NOT use test labels.
        return self

    def transform(self, X):
        # X: numpy array (n_samples, n_features) with NaN.
        # Return: numpy array of the same shape with NO NaNs (finite values).
        return X_imputed
```

Available libraries: `numpy`, `scipy`, `scikit-learn` (all submodules: `sklearn.impute`, `sklearn.ensemble`, `sklearn.neighbors`, ...).

## Fixed Pipeline
The datasets, missingness corruption, and the evaluation pipeline (downstream models and metrics) are fixed by the harness and not editable. Your imputer receives `X` (a numpy array with NaN entries) and must produce a complete array of the same shape with no NaNs. Do not use test labels during imputation.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scikit-learn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits that change code outside these ranges — or creating new files, or
deleting whole files — will cause your submission to be invalid.

The line numbers mark an editable **region**, not a fixed line-count budget: you
may add or remove lines inside it. Only code outside the editable ranges must
stay unchanged.

- `scikit-learn/custom_imputation.py`
- editable lines **36–131**




## Readable Context


### `scikit-learn/custom_imputation.py`  [EDITABLE — lines 36–131 only]

```python
     1: """Custom missing data imputation benchmark.
     2: 
     3: This script evaluates a missing data imputation method across multiple datasets
     4: with artificially introduced missing values. The agent should modify the EDITABLE
     5: section to implement a novel imputation algorithm.
     6: 
     7: Datasets (selected by $ENV):
     8:   - Several standard tabular datasets are used for evaluation (a mix of
     9:     classification and regression); the specific datasets, shapes, and
    10:     train/eval splits are withheld from this file so the imputer must generalize.
    11: 
    12: Missing patterns: MCAR (Missing Completely At Random) at 20% rate.
    13: 
    14: Metrics:
    15:   - rmse:           Root Mean Squared Error of imputed vs true values (lower is better)
    16:   - downstream_score: Classification accuracy or regression R^2 on imputed data (higher is better)
    17: """
    18: 
    19: import os
    20: import sys
    21: import warnings
    22: import numpy as np
    23: # (dataset loaders live in the host-only scoring module; the eval datasets are not named here)
    24: from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    25: from sklearn.preprocessing import StandardScaler
    26: from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    27: from sklearn.metrics import mean_squared_error
    28: from sklearn.base import BaseEstimator, TransformerMixin
    29: 
    30: warnings.filterwarnings("ignore")
    31: 
    32: # ================================================================
    33: # FIXED -- do not modify above this line
    34: # ================================================================
    35: 
    36: # ================================================================
    37: # EDITABLE -- agent modifies this section (lines 36 to 142)
    38: # ================================================================
    39: 
    40: 
    41: class CustomImputer(BaseEstimator, TransformerMixin):
    42:     """Custom missing data imputation algorithm.
    43: 
    44:     Must implement:
    45:         fit(X) -> self              : learn imputation model from X (with NaNs)
    46:         transform(X) -> X_imputed   : impute missing values in X
    47: 
    48:     The algorithm should:
    49:     - Handle both continuous and categorical-like features
    50:     - Preserve the statistical properties of the data
    51:     - Produce accurate imputations that improve downstream task performance
    52:     - Work well across different dataset sizes and feature types
    53: 
    54:     Args:
    55:         random_state: Random seed for reproducibility.
    56:         max_iter: Maximum number of iterations (for iterative methods).
    57: 
    58:     Notes:
    59:         - Input X is a numpy array of shape (n_samples, n_features) with NaN for missing values
    60:         - Output must have the same shape with no NaN values
    61:         - fit() and transform() can be called separately (sklearn convention)
    62:         - Available imports: numpy, scipy, sklearn (all submodules)
    63:     """
    64: 
    65:     def __init__(self, random_state=42, max_iter=10):
    66:         self.random_state = random_state
    67:         self.max_iter = max_iter
    68: 
    69:     def fit(self, X, y=None):
    70:         """Learn the imputation model from data X.
    71: 
    72:         Args:
    73:             X: array of shape (n_samples, n_features) with NaN for missing values
    74:             y: ignored (present for API compatibility)
    75: 
    76:         Returns:
    77:             self
    78:         """
    79:         # Default: compute column means for mean imputation
    80:         self.statistics_ = np.nanmean(X, axis=0)
    81:         return self
    82: 
    83:     def transform(self, X):
    84:         """Impute missing values in X.
    85: 
    86:         Args:
    87:             X: array of shape (n_samples, n_features) with NaN for missing values
    88: 
    89:         Returns:
    90:             X_imputed: array of shape (n_samples, n_features) with no NaN values
    91:         """
    92:         X_imputed = X.copy()
    93:         for j in range(X.shape[1]):
    94:             mask = np.isnan(X_imputed[:, j])
    95:             X_imputed[mask, j] = self.statistics_[j]
    96:         return X_imputed
    97: 
    98:     def fit_transform(self, X, y=None):
    99:         """Fit and transform in one step.
   100: 
   101:         Args:
   102:             X: array of shape (n_samples, n_features) with NaN for missing values
   103:             y: ignored
   104: 
   105:         Returns:
   106:             X_imputed: array of shape (n_samples, n_features) with no NaN values
   107:         """
   108:         return self.fit(X, y).transform(X)
   109: 
   110: 
   111: # Helper functions for the custom imputer (optional, agent may add more)
   112: def compute_feature_correlations(X):
   113:     """Compute pairwise correlations, ignoring NaN pairs.
   114: 
   115:     Args:
   116:         X: array of shape (n_samples, n_features) with possible NaN values
   117: 
   118:     Returns:
   119:         corr: array of shape (n_features, n_features) with correlation coefficients
   120:     """
   121:     n_features = X.shape[1]
   122:     corr = np.eye(n_features)
   123:     for i in range(n_features):
   124:         for j in range(i + 1, n_features):
   125:             mask = ~(np.isnan(X[:, i]) | np.isnan(X[:, j]))
   126:             if mask.sum() > 2:
   127:                 c = np.corrcoef(X[mask, i], X[mask, j])[0, 1]
   128:                 corr[i, j] = corr[j, i] = c if not np.isnan(c) else 0.0
   129:     return corr
   130: 
   131: 
   132: # ================================================================
   133: # FIXED -- input loading + prediction emit (do not modify below this line)
   134: # ================================================================
   135: # The true matrix, the missingness mask, the labels, the dataset identity, and
   136: # the metrics all live in a host-only module the agent's process cannot import.
   137: # This program only loads the pre-generated masked matrix, runs the imputer, and
   138: # emits the imputed matrix; the host-side parser regenerates the truth and
   139: # scores it with the same RMSE + downstream metrics.
   140: 
   141: 
   142: def _impute_inputs_dir():
   143:     d = os.environ.get("IMPUTE_INPUTS_DIR")
   144:     if d:
   145:         return d
   146:     return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_impute_inputs")
   147: 
   148: 
   149: def _load_input(env_name, seed):
   150:     import io as _io
   151:     import base64 as _b64
   152:     path = os.path.join(_impute_inputs_dir(), f"{env_name}_seed{seed}.npy.b64")
   153:     with open(path, "r") as f:
   154:         raw = _b64.b64decode(f.read())
   155:     return np.load(_io.BytesIO(raw))
   156: 
   157: 
   158: def main():
   159:     import base64 as _b64
   160:     env = os.environ.get("ENV", "")
   161:     if not env:
   162:         raise SystemExit("ENV not set")
   163:     seed = int(os.environ.get("SEED", "42"))
   164:     print(f"=== Missing Data Imputation benchmark: {env} (seed={seed}) ===", flush=True)
   165: 
   166:     X_missing = _load_input(env, seed)
   167:     print(f"Input: samples={X_missing.shape[0]}, features={X_missing.shape[1]}", flush=True)
   168: 
   169:     print("TRAIN_METRICS stage=fitting", flush=True)
   170:     imputer = CustomImputer(random_state=seed)
   171:     X_imputed = imputer.fit_transform(X_missing)
   172:     print("TRAIN_METRICS stage=done", flush=True)
   173: 
   174:     X_imputed = np.asarray(X_imputed, dtype=np.float64)
   175:     if np.isnan(X_imputed).any():
   176:         print("WARNING: Imputed data still contains NaN! Filling with column means.", flush=True)
   177:         col_means = np.nanmean(X_imputed, axis=0)
   178:         for j in range(X_imputed.shape[1]):
   179:             nan_mask = np.isnan(X_imputed[:, j])
   180:             X_imputed[nan_mask, j] = col_means[j]
   181: 
   182:     payload = _b64.b64encode(np.ascontiguousarray(X_imputed, dtype=np.float64).tobytes()).decode("ascii")
   183:     print(f"IMPUTE_PRED env={env} seed={seed} rows={X_imputed.shape[0]} cols={X_imputed.shape[1]} "
   184:           f"X_imputed={payload}", flush=True)
   185:     print("Done.", flush=True)
   186: 
   187: 
   188: if __name__ == "__main__":
   189:     main()
```

## Parameter Budget

The check counts trainable parameters in any torch module your imputer attaches, and leaves room for a small learned head — the reference imputers attach none, so that headroom is yours. A heavy deep imputation network (VAE / DAE / MIWAE / diffusion) is out of scope. The check runs automatically — you don't need to invoke it — and going materially beyond that makes the run invalid. The contribution must be algorithmic, not extra capacity.

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `mean_impute` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_imputation.py`:

```python
Lines 36–37:
    33: # FIXED -- do not modify above this line
    34: # ================================================================
    35: 
    36: # ================================================================
    37: # EDITABLE -- agent modifies this section (lines 36 to 142)
    38: # ================================================================
    39: 
    40: 
```

### `knn` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_imputation.py`:

```python
Lines 36–37:
    33: # FIXED -- do not modify above this line
    34: # ================================================================
    35: 
    36: # ================================================================
    37: # EDITABLE -- agent modifies this section (lines 36 to 142)
    38: # ================================================================
    39: 
    40: 
```

### `mice` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_imputation.py`:

```python
Lines 36–37:
    33: # FIXED -- do not modify above this line
    34: # ================================================================
    35: 
    36: # ================================================================
    37: # EDITABLE -- agent modifies this section (lines 36 to 142)
    38: # ================================================================
    39: 
    40: 
```

### `missforest` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_imputation.py`:

```python
Lines 36–37:
    33: # FIXED -- do not modify above this line
    34: # ================================================================
    35: 
    36: # ================================================================
    37: # EDITABLE -- agent modifies this section (lines 36 to 142)
    38: # ================================================================
    39: 
    40: 
```

### `gain` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_imputation.py`:

```python
Lines 36–37:
    33: # FIXED -- do not modify above this line
    34: # ================================================================
    35: 
    36: # ================================================================
    37: # EDITABLE -- agent modifies this section (lines 36 to 142)
    38: # ================================================================
    39: 
    40: 
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
