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
Data is standardized tabular data (classification and regression tasks) with **20% MCAR (Missing Completely At Random)** corruption applied to features. Your imputer receives `X` with NaN entries and must produce a complete array with no NaNs. Do not use test labels during imputation.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scikit-learn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

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
     8:   - breast_cancer:  Classification, 569 samples x 30 features (binary)
     9:   - wine:           Classification, 178 samples x 13 features (3-class)
    10:   - california:     Regression, 20640 samples x 8 features (continuous target)
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
    23: from sklearn.datasets import load_breast_cancer, load_wine, fetch_california_housing
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
   133: # FIXED -- do not modify below this line
   134: # ================================================================
   135: 
   136: 
   137: def load_dataset(env_name, seed=42):
   138:     """Load dataset and return X, y, and task type."""
   139:     if env_name == "breast_cancer":
   140:         data = load_breast_cancer()
   141:         return data.data, data.target, "classification"
   142:     elif env_name == "wine":
   143:         data = load_wine()
   144:         return data.data, data.target, "classification"
   145:     elif env_name == "california":
   146:         data = fetch_california_housing()
   147:         # Subsample for speed (use first 5000 samples)
   148:         rng = np.random.RandomState(seed)
   149:         idx = rng.choice(len(data.data), min(5000, len(data.data)), replace=False)
   150:         return data.data[idx], data.target[idx], "regression"
   151:     else:
   152:         raise ValueError(f"Unknown environment: {env_name}")
   153: 
   154: 
   155: def introduce_missing(X, missing_rate=0.20, seed=42):
   156:     """Introduce MCAR missing values at the given rate.
   157: 
   158:     Returns:
   159:         X_missing: array with NaN for missing values
   160:         mask: boolean array, True where values were made missing
   161:     """
   162:     rng = np.random.RandomState(seed)
   163:     mask = rng.random(X.shape) < missing_rate
   164:     # Don't make entire rows or columns missing
   165:     for i in range(X.shape[0]):
   166:         if mask[i].all():
   167:             mask[i, rng.randint(X.shape[1])] = False
   168:     for j in range(X.shape[1]):
   169:         if mask[:, j].all():
   170:             mask[rng.randint(X.shape[0]), j] = False
   171:     X_missing = X.copy()
   172:     X_missing[mask] = np.nan
   173:     return X_missing, mask
   174: 
   175: 
   176: def compute_imputation_rmse(X_true, X_imputed, mask):
   177:     """Compute RMSE only on the artificially missing entries."""
   178:     true_vals = X_true[mask]
   179:     imputed_vals = X_imputed[mask]
   180:     return np.sqrt(mean_squared_error(true_vals, imputed_vals))
   181: 
   182: 
   183: def compute_downstream_score(X_imputed, y, task_type, seed=42):
   184:     """Compute downstream predictive performance using cross-validation."""
   185:     if task_type == "classification":
   186:         model = GradientBoostingClassifier(
   187:             n_estimators=100, max_depth=3, random_state=seed
   188:         )
   189:         cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
   190:         scores = cross_val_score(model, X_imputed, y, cv=cv, scoring="accuracy")
   191:     else:
   192:         model = GradientBoostingRegressor(
   193:             n_estimators=100, max_depth=3, random_state=seed
   194:         )
   195:         cv = KFold(n_splits=5, shuffle=True, random_state=seed)
   196:         scores = cross_val_score(model, X_imputed, y, cv=cv, scoring="r2")
   197:     return scores.mean()
   198: 
   199: 
   200: def main():
   201:     env = os.environ.get("ENV", "breast_cancer")
   202:     seed = int(os.environ.get("SEED", "42"))
   203: 
   204:     print(f"=== Missing Data Imputation benchmark: {env} (seed={seed}) ===", flush=True)
   205: 
   206:     # Load data
   207:     X_raw, y, task_type = load_dataset(env, seed=seed)
   208: 
   209:     # Standardize features (on full data, before introducing missing values)
   210:     scaler = StandardScaler()
   211:     X_scaled = scaler.fit_transform(X_raw)
   212: 
   213:     print(
   214:         f"Dataset: {env}, samples={X_scaled.shape[0]}, features={X_scaled.shape[1]}, "
   215:         f"task={task_type}",
   216:         flush=True,
   217:     )
   218: 
   219:     # Introduce missing values (MCAR at 20%)
   220:     X_missing, mask = introduce_missing(X_scaled, missing_rate=0.20, seed=seed)
   221:     n_missing = mask.sum()
   222:     print(
   223:         f"Missing entries: {n_missing} / {X_scaled.size} "
   224:         f"({100 * n_missing / X_scaled.size:.1f}%)",
   225:         flush=True,
   226:     )
   227: 
   228:     # Run custom imputer
   229:     print("TRAIN_METRICS stage=fitting", flush=True)
   230:     imputer = CustomImputer(random_state=seed)
   231:     X_imputed = imputer.fit_transform(X_missing)
   232:     print("TRAIN_METRICS stage=done", flush=True)
   233: 
   234:     # Check for remaining NaN
   235:     if np.isnan(X_imputed).any():
   236:         print("WARNING: Imputed data still contains NaN! Filling with column means.", flush=True)
   237:         col_means = np.nanmean(X_imputed, axis=0)
   238:         for j in range(X_imputed.shape[1]):
   239:             nan_mask = np.isnan(X_imputed[:, j])
   240:             X_imputed[nan_mask, j] = col_means[j]
   241: 
   242:     # Compute imputation RMSE
   243:     rmse = compute_imputation_rmse(X_scaled, X_imputed, mask)
   244:     print(f"TRAIN_METRICS rmse={rmse:.6f}", flush=True)
   245: 
   246:     # Compute downstream score
   247:     downstream = compute_downstream_score(X_imputed, y, task_type, seed=seed)
   248:     print(f"TRAIN_METRICS downstream_score={downstream:.6f}", flush=True)
   249: 
   250:     # Also compute baseline (no missing data) downstream score for reference
   251:     baseline_score = compute_downstream_score(X_scaled, y, task_type, seed=seed)
   252:     print(f"TRAIN_METRICS baseline_no_missing={baseline_score:.6f}", flush=True)
   253: 
   254:     # Final metrics
   255:     print(f"TEST_METRICS rmse={rmse:.6f} downstream_score={downstream:.6f}", flush=True)
   256: 
   257:     print("Done.", flush=True)
   258: 
   259: 
   260: if __name__ == "__main__":
   261:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `mean_impute` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_imputation.py`:

```python
Lines 36–73:
    33: # FIXED -- do not modify above this line
    34: # ================================================================
    35: 
    36: # ================================================================
    37: # EDITABLE -- agent modifies this section (lines 36 to 142)
    38: # ================================================================
    39: 
    40: 
    41: class CustomImputer(BaseEstimator, TransformerMixin):
    42:     """Mean Imputation: replace missing values with column means."""
    43: 
    44:     def __init__(self, random_state=42, max_iter=10):
    45:         self.random_state = random_state
    46:         self.max_iter = max_iter
    47: 
    48:     def fit(self, X, y=None):
    49:         self.statistics_ = np.nanmean(X, axis=0)
    50:         return self
    51: 
    52:     def transform(self, X):
    53:         X_imputed = X.copy()
    54:         for j in range(X.shape[1]):
    55:             mask = np.isnan(X_imputed[:, j])
    56:             X_imputed[mask, j] = self.statistics_[j]
    57:         return X_imputed
    58: 
    59:     def fit_transform(self, X, y=None):
    60:         return self.fit(X, y).transform(X)
    61: 
    62: 
    63: def compute_feature_correlations(X):
    64:     n_features = X.shape[1]
    65:     corr = np.eye(n_features)
    66:     for i in range(n_features):
    67:         for j in range(i + 1, n_features):
    68:             mask = ~(np.isnan(X[:, i]) | np.isnan(X[:, j]))
    69:             if mask.sum() > 2:
    70:                 c = np.corrcoef(X[mask, i], X[mask, j])[0, 1]
    71:                 corr[i, j] = corr[j, i] = c if not np.isnan(c) else 0.0
    72:     return corr
    73: 
    74: # ================================================================
    75: # FIXED -- do not modify below this line
    76: # ================================================================
```

### `knn` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_imputation.py`:

```python
Lines 36–84:
    33: # FIXED -- do not modify above this line
    34: # ================================================================
    35: 
    36: # ================================================================
    37: # EDITABLE -- agent modifies this section (lines 36 to 142)
    38: # ================================================================
    39: 
    40: 
    41: class CustomImputer(BaseEstimator, TransformerMixin):
    42:     """KNN Imputation: impute using K-nearest neighbors.
    43: 
    44:     Uses sklearn.impute.KNNImputer with n_neighbors=5, distance weighting.
    45:     Reference: Troyanskaya et al. (2001).
    46:     """
    47: 
    48:     def __init__(self, random_state=42, max_iter=10):
    49:         self.random_state = random_state
    50:         self.max_iter = max_iter
    51:         self.n_neighbors = 5
    52: 
    53:     def fit(self, X, y=None):
    54:         from sklearn.impute import KNNImputer
    55:         self._imputer = KNNImputer(
    56:             n_neighbors=self.n_neighbors,
    57:             weights="distance",
    58:         )
    59:         self._imputer.fit(X)
    60:         return self
    61: 
    62:     def transform(self, X):
    63:         return self._imputer.transform(X)
    64: 
    65:     def fit_transform(self, X, y=None):
    66:         from sklearn.impute import KNNImputer
    67:         self._imputer = KNNImputer(
    68:             n_neighbors=self.n_neighbors,
    69:             weights="distance",
    70:         )
    71:         return self._imputer.fit_transform(X)
    72: 
    73: 
    74: def compute_feature_correlations(X):
    75:     n_features = X.shape[1]
    76:     corr = np.eye(n_features)
    77:     for i in range(n_features):
    78:         for j in range(i + 1, n_features):
    79:             mask = ~(np.isnan(X[:, i]) | np.isnan(X[:, j]))
    80:             if mask.sum() > 2:
    81:                 c = np.corrcoef(X[mask, i], X[mask, j])[0, 1]
    82:                 corr[i, j] = corr[j, i] = c if not np.isnan(c) else 0.0
    83:     return corr
    84: 
    85: # ================================================================
    86: # FIXED -- do not modify below this line
    87: # ================================================================
```

### `mice` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_imputation.py`:

```python
Lines 36–97:
    33: # FIXED -- do not modify above this line
    34: # ================================================================
    35: 
    36: # ================================================================
    37: # EDITABLE -- agent modifies this section (lines 36 to 142)
    38: # ================================================================
    39: 
    40: 
    41: class CustomImputer(BaseEstimator, TransformerMixin):
    42:     """MICE: Multiple Imputation by Chained Equations.
    43: 
    44:     Uses sklearn.impute.IterativeImputer with BayesianRidge estimator.
    45:     Reference: van Buuren & Groothuis-Oudshoorn (2011).
    46:     """
    47: 
    48:     def __init__(self, random_state=42, max_iter=30):
    49:         self.random_state = random_state
    50:         self.max_iter = max_iter
    51: 
    52:     def fit(self, X, y=None):
    53:         from sklearn.experimental import enable_iterative_imputer  # noqa
    54:         from sklearn.impute import IterativeImputer
    55:         from sklearn.linear_model import BayesianRidge
    56: 
    57:         self._imputer = IterativeImputer(
    58:             estimator=BayesianRidge(),
    59:             max_iter=self.max_iter,
    60:             random_state=self.random_state,
    61:             imputation_order="ascending",
    62:             initial_strategy="mean",
    63:             tol=1e-3,
    64:         )
    65:         self._imputer.fit(X)
    66:         return self
    67: 
    68:     def transform(self, X):
    69:         return self._imputer.transform(X)
    70: 
    71:     def fit_transform(self, X, y=None):
    72:         from sklearn.experimental import enable_iterative_imputer  # noqa
    73:         from sklearn.impute import IterativeImputer
    74:         from sklearn.linear_model import BayesianRidge
    75: 
    76:         self._imputer = IterativeImputer(
    77:             estimator=BayesianRidge(),
    78:             max_iter=self.max_iter,
    79:             random_state=self.random_state,
    80:             imputation_order="ascending",
    81:             initial_strategy="mean",
    82:             tol=1e-3,
    83:         )
    84:         return self._imputer.fit_transform(X)
    85: 
    86: 
    87: def compute_feature_correlations(X):
    88:     n_features = X.shape[1]
    89:     corr = np.eye(n_features)
    90:     for i in range(n_features):
    91:         for j in range(i + 1, n_features):
    92:             mask = ~(np.isnan(X[:, i]) | np.isnan(X[:, j]))
    93:             if mask.sum() > 2:
    94:                 c = np.corrcoef(X[mask, i], X[mask, j])[0, 1]
    95:                 corr[i, j] = corr[j, i] = c if not np.isnan(c) else 0.0
    96:     return corr
    97: 
    98: # ================================================================
    99: # FIXED -- do not modify below this line
   100: # ================================================================
```

### `missforest` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_imputation.py`:

```python
Lines 36–145:
    33: # FIXED -- do not modify above this line
    34: # ================================================================
    35: 
    36: # ================================================================
    37: # EDITABLE -- agent modifies this section (lines 36 to 142)
    38: # ================================================================
    39: 
    40: 
    41: class CustomImputer(BaseEstimator, TransformerMixin):
    42:     """MissForest: Iterative Random Forest imputation.
    43: 
    44:     Implements the MissForest algorithm (Stekhoven & Buehlmann, 2012):
    45:     1. Initial imputation with column means
    46:     2. For each iteration:
    47:        a. Sort features by missingness (ascending)
    48:        b. For each feature with missing values:
    49:           - Train RandomForest on observed entries using all other features
    50:           - Predict missing entries
    51:        c. Check convergence (normalized difference < tol)
    52:     3. Return when converged or max_iter reached
    53: 
    54:     Reference: Bioinformatics 28(1):112-118, 2012.
    55:     """
    56: 
    57:     def __init__(self, random_state=42, max_iter=10):
    58:         self.random_state = random_state
    59:         self.max_iter = max_iter
    60:         self.n_estimators = 100
    61:         self.tol = 1e-4
    62: 
    63:     def fit(self, X, y=None):
    64:         # Store the fitted state by running fit_transform internally
    65:         self._X_fitted = X.copy()
    66:         self._fit_transform_internal(X)
    67:         return self
    68: 
    69:     def transform(self, X):
    70:         return self._fit_transform_internal(X)
    71: 
    72:     def fit_transform(self, X, y=None):
    73:         return self._fit_transform_internal(X)
    74: 
    75:     def _fit_transform_internal(self, X):
    76:         from sklearn.ensemble import RandomForestRegressor
    77: 
    78:         X_imp = X.copy()
    79:         n_samples, n_features = X_imp.shape
    80: 
    81:         # Step 1: Initial imputation with column means
    82:         col_means = np.nanmean(X_imp, axis=0)
    83:         for j in range(n_features):
    84:             mask_j = np.isnan(X_imp[:, j])
    85:             X_imp[mask_j, j] = col_means[j]
    86: 
    87:         # Identify which features have missing values and sort by missingness
    88:         miss_count = np.isnan(X).sum(axis=0)
    89:         features_with_missing = np.where(miss_count > 0)[0]
    90:         # Sort by number of missing values (ascending)
    91:         features_with_missing = features_with_missing[
    92:             np.argsort(miss_count[features_with_missing])
    93:         ]
    94: 
    95:         if len(features_with_missing) == 0:
    96:             return X_imp
    97: 
    98:         # Step 2: Iterative imputation
    99:         for iteration in range(self.max_iter):
   100:             X_prev = X_imp.copy()
   101: 
   102:             for j in features_with_missing:
   103:                 # Observed and missing indices for feature j
   104:                 obs_mask = ~np.isnan(X[:, j])
   105:                 mis_mask = np.isnan(X[:, j])
   106: 
   107:                 if mis_mask.sum() == 0:
   108:                     continue
   109: 
   110:                 # Predictor features (all except j)
   111:                 other_features = [k for k in range(n_features) if k != j]
   112:                 X_train = X_imp[obs_mask][:, other_features]
   113:                 y_train = X[obs_mask, j]  # Use original observed values
   114:                 X_pred = X_imp[mis_mask][:, other_features]
   115: 
   116:                 # Train random forest and predict
   117:                 rf = RandomForestRegressor(
   118:                     n_estimators=self.n_estimators,
   119:                     max_features="sqrt",
   120:                     random_state=self.random_state,
   121:                     n_jobs=-1,
   122:                 )
   123:                 rf.fit(X_train, y_train)
   124:                 X_imp[mis_mask, j] = rf.predict(X_pred)
   125: 
   126:             # Step 3: Check convergence
   127:             diff = np.sum((X_imp - X_prev) ** 2)
   128:             denom = np.sum(X_imp ** 2)
   129:             if denom > 0 and diff / denom < self.tol:
   130:                 break
   131: 
   132:         return X_imp
   133: 
   134: 
   135: def compute_feature_correlations(X):
   136:     n_features = X.shape[1]
   137:     corr = np.eye(n_features)
   138:     for i in range(n_features):
   139:         for j in range(i + 1, n_features):
   140:             mask = ~(np.isnan(X[:, i]) | np.isnan(X[:, j]))
   141:             if mask.sum() > 2:
   142:                 c = np.corrcoef(X[mask, i], X[mask, j])[0, 1]
   143:                 corr[i, j] = corr[j, i] = c if not np.isnan(c) else 0.0
   144:     return corr
   145: 
   146: # ================================================================
   147: # FIXED -- do not modify below this line
   148: # ================================================================
```

### `gain` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_imputation.py`:

```python
Lines 36–102:
    33: # FIXED -- do not modify above this line
    34: # ================================================================
    35: 
    36: # ================================================================
    37: # EDITABLE -- agent modifies this section (lines 36 to 142)
    38: # ================================================================
    39: 
    40: 
    41: class CustomImputer(BaseEstimator, TransformerMixin):
    42:     """Iterative imputation with ExtraTreesRegressor.
    43: 
    44:     Uses sklearn's IterativeImputer with ExtraTreesRegressor as the
    45:     estimator. ExtraTrees captures non-linear feature dependencies
    46:     (similar to GAIN's goal) but converges reliably. Each feature
    47:     with missing values is modeled as a function of all other features,
    48:     iterated in round-robin until convergence.
    49: 
    50:     This replaces the original numpy GAIN (GAN) baseline which could
    51:     not converge due to incomplete backpropagation.
    52:     """
    53: 
    54:     def __init__(self, random_state=42, max_iter=10):
    55:         self.random_state = random_state
    56:         self.max_iter = max_iter
    57:         self.n_estimators = 100
    58: 
    59:     def _make_imputer(self):
    60:         from sklearn.experimental import enable_iterative_imputer  # noqa
    61:         from sklearn.impute import IterativeImputer
    62:         from sklearn.ensemble import ExtraTreesRegressor
    63: 
    64:         estimator = ExtraTreesRegressor(
    65:             n_estimators=self.n_estimators,
    66:             max_features="sqrt",
    67:             random_state=self.random_state,
    68:             n_jobs=-1,
    69:         )
    70:         return IterativeImputer(
    71:             estimator=estimator,
    72:             max_iter=self.max_iter,
    73:             random_state=self.random_state,
    74:             imputation_order="ascending",
    75:             initial_strategy="mean",
    76:             tol=1e-3,
    77:         )
    78: 
    79:     def fit(self, X, y=None):
    80:         self._imputer = self._make_imputer()
    81:         self._imputer.fit(X)
    82:         return self
    83: 
    84:     def transform(self, X):
    85:         return self._imputer.transform(X)
    86: 
    87:     def fit_transform(self, X, y=None):
    88:         self._imputer = self._make_imputer()
    89:         return self._imputer.fit_transform(X)
    90: 
    91: 
    92: def compute_feature_correlations(X):
    93:     n_features = X.shape[1]
    94:     corr = np.eye(n_features)
    95:     for i in range(n_features):
    96:         for j in range(i + 1, n_features):
    97:             mask = ~(np.isnan(X[:, i]) | np.isnan(X[:, j]))
    98:             if mask.sum() > 2:
    99:                 c = np.corrcoef(X[mask, i], X[mask, j])[0, 1]
   100:                 corr[i, j] = corr[j, i] = c if not np.isnan(c) else 0.0
   101:     return corr
   102: 
   103: # ================================================================
   104: # FIXED -- do not modify below this line
   105: # ================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
