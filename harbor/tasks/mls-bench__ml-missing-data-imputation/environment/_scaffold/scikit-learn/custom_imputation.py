"""Custom missing data imputation benchmark.

This script evaluates a missing data imputation method across multiple datasets
with artificially introduced missing values. The agent should modify the EDITABLE
section to implement a novel imputation algorithm.

Datasets (selected by $ENV):
  - breast_cancer:  Classification, 569 samples x 30 features (binary)
  - wine:           Classification, 178 samples x 13 features (3-class)
  - california:     Regression, 20640 samples x 8 features (continuous target)

Missing patterns: MCAR (Missing Completely At Random) at 20% rate.

Metrics:
  - rmse:           Root Mean Squared Error of imputed vs true values (lower is better)
  - downstream_score: Classification accuracy or regression R^2 on imputed data (higher is better)
"""

import os
import sys
import warnings
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, fetch_california_housing
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
# FIXED -- do not modify below this line
# ================================================================


def load_dataset(env_name, seed=42):
    """Load dataset and return X, y, and task type."""
    if env_name == "breast_cancer":
        data = load_breast_cancer()
        return data.data, data.target, "classification"
    elif env_name == "wine":
        data = load_wine()
        return data.data, data.target, "classification"
    elif env_name == "california":
        data = fetch_california_housing()
        # Subsample for speed (use first 5000 samples)
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(data.data), min(5000, len(data.data)), replace=False)
        return data.data[idx], data.target[idx], "regression"
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def introduce_missing(X, missing_rate=0.20, seed=42):
    """Introduce MCAR missing values at the given rate.

    Returns:
        X_missing: array with NaN for missing values
        mask: boolean array, True where values were made missing
    """
    rng = np.random.RandomState(seed)
    mask = rng.random(X.shape) < missing_rate
    # Don't make entire rows or columns missing
    for i in range(X.shape[0]):
        if mask[i].all():
            mask[i, rng.randint(X.shape[1])] = False
    for j in range(X.shape[1]):
        if mask[:, j].all():
            mask[rng.randint(X.shape[0]), j] = False
    X_missing = X.copy()
    X_missing[mask] = np.nan
    return X_missing, mask


def compute_imputation_rmse(X_true, X_imputed, mask):
    """Compute RMSE only on the artificially missing entries."""
    true_vals = X_true[mask]
    imputed_vals = X_imputed[mask]
    return np.sqrt(mean_squared_error(true_vals, imputed_vals))


def compute_downstream_score(X_imputed, y, task_type, seed=42):
    """Compute downstream predictive performance using cross-validation."""
    if task_type == "classification":
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=seed
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        scores = cross_val_score(model, X_imputed, y, cv=cv, scoring="accuracy")
    else:
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=seed
        )
        cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        scores = cross_val_score(model, X_imputed, y, cv=cv, scoring="r2")
    return scores.mean()


def main():
    env = os.environ.get("ENV", "breast_cancer")
    seed = int(os.environ.get("SEED", "42"))

    print(f"=== Missing Data Imputation benchmark: {env} (seed={seed}) ===", flush=True)

    # Load data
    X_raw, y, task_type = load_dataset(env, seed=seed)

    # Standardize features (on full data, before introducing missing values)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    print(
        f"Dataset: {env}, samples={X_scaled.shape[0]}, features={X_scaled.shape[1]}, "
        f"task={task_type}",
        flush=True,
    )

    # Introduce missing values (MCAR at 20%)
    X_missing, mask = introduce_missing(X_scaled, missing_rate=0.20, seed=seed)
    n_missing = mask.sum()
    print(
        f"Missing entries: {n_missing} / {X_scaled.size} "
        f"({100 * n_missing / X_scaled.size:.1f}%)",
        flush=True,
    )

    # Run custom imputer
    print("TRAIN_METRICS stage=fitting", flush=True)
    imputer = CustomImputer(random_state=seed)
    X_imputed = imputer.fit_transform(X_missing)
    print("TRAIN_METRICS stage=done", flush=True)

    # Check for remaining NaN
    if np.isnan(X_imputed).any():
        print("WARNING: Imputed data still contains NaN! Filling with column means.", flush=True)
        col_means = np.nanmean(X_imputed, axis=0)
        for j in range(X_imputed.shape[1]):
            nan_mask = np.isnan(X_imputed[:, j])
            X_imputed[nan_mask, j] = col_means[j]

    # Compute imputation RMSE
    rmse = compute_imputation_rmse(X_scaled, X_imputed, mask)
    print(f"TRAIN_METRICS rmse={rmse:.6f}", flush=True)

    # Compute downstream score
    downstream = compute_downstream_score(X_imputed, y, task_type, seed=seed)
    print(f"TRAIN_METRICS downstream_score={downstream:.6f}", flush=True)

    # Also compute baseline (no missing data) downstream score for reference
    baseline_score = compute_downstream_score(X_scaled, y, task_type, seed=seed)
    print(f"TRAIN_METRICS baseline_no_missing={baseline_score:.6f}", flush=True)

    # Final metrics
    print(f"TEST_METRICS rmse={rmse:.6f} downstream_score={downstream:.6f}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
