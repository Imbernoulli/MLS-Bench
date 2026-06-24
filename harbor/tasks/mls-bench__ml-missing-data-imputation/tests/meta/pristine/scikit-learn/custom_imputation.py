"""Custom missing data imputation benchmark.

This script evaluates a missing data imputation method across multiple datasets
with artificially introduced missing values. The agent should modify the EDITABLE
section to implement a novel imputation algorithm.

Datasets (selected by $ENV):
  - Several standard tabular datasets are used for evaluation (a mix of
    classification and regression); the specific datasets, shapes, and
    train/eval splits are withheld from this file so the imputer must generalize.

Missing patterns: MCAR (Missing Completely At Random) at 20% rate.

Metrics:
  - rmse:           Root Mean Squared Error of imputed vs true values (lower is better)
  - downstream_score: Classification accuracy or regression R^2 on imputed data (higher is better)
"""

import os
import sys
import warnings
import numpy as np
# (dataset loaders live in the host-only scoring module; the eval datasets are not named here)
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")

# ================================================================
# FIXED -- do not modify above this line
# ================================================================

# ================================================================
# EDITABLE -- agent modifies this section (lines 36 to 142)
# ================================================================


class CustomImputer(BaseEstimator, TransformerMixin):
    """Custom missing data imputation algorithm.

    Must implement:
        fit(X) -> self              : learn imputation model from X (with NaNs)
        transform(X) -> X_imputed   : impute missing values in X

    The algorithm should:
    - Handle both continuous and categorical-like features
    - Preserve the statistical properties of the data
    - Produce accurate imputations that improve downstream task performance
    - Work well across different dataset sizes and feature types

    Args:
        random_state: Random seed for reproducibility.
        max_iter: Maximum number of iterations (for iterative methods).

    Notes:
        - Input X is a numpy array of shape (n_samples, n_features) with NaN for missing values
        - Output must have the same shape with no NaN values
        - fit() and transform() can be called separately (sklearn convention)
        - Available imports: numpy, scipy, sklearn (all submodules)
    """

    def __init__(self, random_state=42, max_iter=10):
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X, y=None):
        """Learn the imputation model from data X.

        Args:
            X: array of shape (n_samples, n_features) with NaN for missing values
            y: ignored (present for API compatibility)

        Returns:
            self
        """
        # Default: compute column means for mean imputation
        self.statistics_ = np.nanmean(X, axis=0)
        return self

    def transform(self, X):
        """Impute missing values in X.

        Args:
            X: array of shape (n_samples, n_features) with NaN for missing values

        Returns:
            X_imputed: array of shape (n_samples, n_features) with no NaN values
        """
        X_imputed = X.copy()
        for j in range(X.shape[1]):
            mask = np.isnan(X_imputed[:, j])
            X_imputed[mask, j] = self.statistics_[j]
        return X_imputed

    def fit_transform(self, X, y=None):
        """Fit and transform in one step.

        Args:
            X: array of shape (n_samples, n_features) with NaN for missing values
            y: ignored

        Returns:
            X_imputed: array of shape (n_samples, n_features) with no NaN values
        """
        return self.fit(X, y).transform(X)


# Helper functions for the custom imputer (optional, agent may add more)
def compute_feature_correlations(X):
    """Compute pairwise correlations, ignoring NaN pairs.

    Args:
        X: array of shape (n_samples, n_features) with possible NaN values

    Returns:
        corr: array of shape (n_features, n_features) with correlation coefficients
    """
    n_features = X.shape[1]
    corr = np.eye(n_features)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            mask = ~(np.isnan(X[:, i]) | np.isnan(X[:, j]))
            if mask.sum() > 2:
                c = np.corrcoef(X[mask, i], X[mask, j])[0, 1]
                corr[i, j] = corr[j, i] = c if not np.isnan(c) else 0.0
    return corr


# ================================================================
# FIXED -- input loading + prediction emit (do not modify below this line)
# ================================================================
# The true matrix, the missingness mask, the labels, the dataset identity, and
# the metrics all live in a host-only module the agent's process cannot import.
# This program only loads the pre-generated masked matrix, runs the imputer, and
# emits the imputed matrix; the host-side parser regenerates the truth and
# scores it with the same RMSE + downstream metrics.


def _impute_inputs_dir():
    d = os.environ.get("IMPUTE_INPUTS_DIR")
    if d:
        return d
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_impute_inputs")


def _load_input(env_name, seed):
    import io as _io
    import base64 as _b64
    path = os.path.join(_impute_inputs_dir(), f"{env_name}_seed{seed}.npy.b64")
    with open(path, "r") as f:
        raw = _b64.b64decode(f.read())
    return np.load(_io.BytesIO(raw))


def main():
    import base64 as _b64
    env = os.environ.get("ENV", "")
    if not env:
        raise SystemExit("ENV not set")
    seed = int(os.environ.get("SEED", "42"))
    print(f"=== Missing Data Imputation benchmark: {env} (seed={seed}) ===", flush=True)

    X_missing = _load_input(env, seed)
    print(f"Input: samples={X_missing.shape[0]}, features={X_missing.shape[1]}", flush=True)

    print("TRAIN_METRICS stage=fitting", flush=True)
    imputer = CustomImputer(random_state=seed)
    X_imputed = imputer.fit_transform(X_missing)
    print("TRAIN_METRICS stage=done", flush=True)

    X_imputed = np.asarray(X_imputed, dtype=np.float64)
    if np.isnan(X_imputed).any():
        print("WARNING: Imputed data still contains NaN! Filling with column means.", flush=True)
        col_means = np.nanmean(X_imputed, axis=0)
        for j in range(X_imputed.shape[1]):
            nan_mask = np.isnan(X_imputed[:, j])
            X_imputed[nan_mask, j] = col_means[j]

    payload = _b64.b64encode(np.ascontiguousarray(X_imputed, dtype=np.float64).tobytes()).decode("ascii")
    print(f"IMPUTE_PRED env={env} seed={seed} rows={X_imputed.shape[0]} cols={X_imputed.shape[1]} "
          f"X_imputed={payload}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
