# MLS-Bench: ml-missing-data-imputation

# Missing Data Imputation

## Research Question
Design a tabular missing-data imputation method that achieves low reconstruction error and preserves downstream predictive performance across diverse datasets. The contribution is the *imputer itself*: how feature dependencies are exploited, how imputations are iterated/refined, and how completed values are produced from data containing NaNs.

## Background
Missing data is ubiquitous. Mean/median imputation ignores feature correlations; iterative predictive methods exploit them.

## Implementation Contract
Implement `CustomImputer` in `scikit-learn/custom_imputation.py`:

```python
class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=42, max_iter=10):
        ...

    def fit(self, X, y=None):
        # X: numpy array (n_samples, n_features) with NaN for missing values.
        # Learn imputation model.
        return self

    def transform(self, X):
        # X: numpy array (n_samples, n_features) with NaN.
        # Return: numpy array of the same shape with NO NaNs (finite values).
        return X_imputed
```

Available libraries: `numpy`, `scipy`, `scikit-learn` (all submodules: `sklearn.impute`, `sklearn.ensemble`, `sklearn.neighbors`, ...).

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scikit-learn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will be rejected.

- `scikit-learn/custom_imputation.py`
- editable lines **36–131**

## Readable Context

### `scikit-learn/custom_imputation.py`  [EDITABLE — lines 36–131 only]

```python
     1: """Custom missing data imputation scaffold."""
     2:
     3: import os
     4: import sys
     5: import warnings
     6: import numpy as np
     7: from sklearn.base import BaseEstimator, TransformerMixin
     8:
     9: warnings.filterwarnings("ignore")
    10:
    11: # ================================================================
    12: # FIXED -- do not modify above this line
    13: # ================================================================
    14:
    15: # ================================================================
    16: # EDITABLE -- agent modifies this section (lines 36 to 142)
    17: # ================================================================
    18:
    19:
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
    70:         self.statistics_ = np.nanmean(X, axis=0)
    71:         return self
    72:
    73:     def transform(self, X):
    74:         X_imputed = X.copy()
    75:         for j in range(X.shape[1]):
    76:             mask = np.isnan(X_imputed[:, j])
    77:             X_imputed[mask, j] = self.statistics_[j]
    78:         return X_imputed
    79:
    80:     def fit_transform(self, X, y=None):
    81:         return self.fit(X, y).transform(X)
    82:
    83:
    84: # Helper functions for the custom imputer (optional, agent may add more)
    85: def compute_feature_correlations(X):
    86:     n_features = X.shape[1]
    87:     corr = np.eye(n_features)
    88:     for i in range(n_features):
    89:         for j in range(i + 1, n_features):
    90:             mask = ~(np.isnan(X[:, i]) | np.isnan(X[:, j]))
    91:             if mask.sum() > 2:
    92:                 c = np.corrcoef(X[mask, i], X[mask, j])[0, 1]
    93:                 corr[i, j] = corr[j, i] = c if not np.isnan(c) else 0.0
    94:     return corr
   131: # ================================================================
```

## Tips

- Keep the function/class signatures of the editable regions identical.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.

Good luck.
