"""Held-out scoring module for ml-missing-data-imputation (NOT agent-visible).

Lives outside every path bind-mounted into the agent container; imported only
by the host-side parser.py (native) / Harbor verifier. Holds the data loader,
the missingness mask, and the metrics — so the editable imputer cannot
regenerate the true matrix to fill the masked entries.

Bodies are byte-identical to the originals that used to live in the editable
template, so honest results are reproduced exactly.
"""

import os
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import mean_squared_error


def _sklearn_data_home():
    # This module only ever runs host-side / in the verifier. Inside the task
    # container SKLEARN_DATA_HOME points at the baked sklearn cache (compute
    # nodes have no network); prefer that env var, else the committed host data
    # dir, else sklearn's default cache. Matches ml-ensemble-boosting's dgp.
    home = os.environ.get("SKLEARN_DATA_HOME")
    if home and Path(home).is_dir():
        return home
    cand = Path(__file__).resolve().parents[2] / "vendor" / "data" / "sklearn"
    if cand.is_dir():
        return str(cand)
    return home


def load_dataset(env_name, seed=42):
    """Load dataset and return X, y, and task type."""
    if env_name == "breast_cancer":
        data = load_breast_cancer()
        return data.data, data.target, "classification"
    elif env_name == "wine":
        data = load_wine()
        return data.data, data.target, "classification"
    elif env_name == "california":
        data = fetch_california_housing(data_home=_sklearn_data_home())
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


# ---- helpers used by the input pre-generator and the host-side scorer ----

def gen_input(env_name, seed=42):
    """Return ONLY the agent-visible input: the standardized matrix with masked
    entries set to NaN. The true values at masked entries are NOT recoverable
    from this (the agent must impute them)."""
    X_raw, _y, _task = load_dataset(env_name, seed=seed)
    X_scaled = StandardScaler().fit_transform(X_raw)
    X_missing, _mask = introduce_missing(X_scaled, missing_rate=0.20, seed=seed)
    return X_missing


def truth(env_name, seed=42):
    """Return held-out ground truth for scoring: (X_scaled, y, mask, task_type)."""
    X_raw, y, task_type = load_dataset(env_name, seed=seed)
    X_scaled = StandardScaler().fit_transform(X_raw)
    _X_missing, mask = introduce_missing(X_scaled, missing_rate=0.20, seed=seed)
    return X_scaled, y, mask, task_type


def opaque_label(label):
    """Host-only opaque token for a dataset label (the agent must not learn which
    dataset is evaluated). Used to name the pre-generated input files and as the
    identity the run scripts pass, so listing the workspace reveals only tokens;
    the real label stays host-side (the scorer uses the test-command label)."""
    import hashlib
    return hashlib.sha1(
        ("ml-missing-data-imputation::dataset-token::v1::" + str(label)).encode("utf-8")
    ).hexdigest()[:12]
