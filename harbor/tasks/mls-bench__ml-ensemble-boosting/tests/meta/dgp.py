"""Held-out scoring module for ml-ensemble-boosting (NOT agent-visible).

Imported only by the host-side parser.py (native) / Harbor verifier. Holds the
dataset identity (the sklearn dataset mapping), the loader, the deterministic
train/test split, the feature normalization, the test labels, and the metric --
so the editable BoostingStrategy can neither name the eval datasets nor recover
y_test (it cannot reload the public data, replay the split, and emit the labels).

Loader / split / normalization / metric bodies are byte-identical to the
originals, so honest results are reproduced exactly.
"""

import os
from pathlib import Path

import numpy as np
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# env label -> (sklearn dataset name, task type)
DATASET_SPECS = {
    "breast_cancer": ("breast_cancer", "classification"),
    "diabetes": ("diabetes", "regression"),
    "california_housing": ("california_housing", "regression"),
}

# Fixed pipeline hyperparameters (same as the original scripts).
N_ROUNDS = 200
MAX_DEPTH = 3
LEARNING_RATE = 0.1
TEST_SIZE = 0.2


def _sklearn_data_home():
    # This module only ever runs host-side. Inside the container SKLEARN_DATA_HOME
    # points at /data/sklearn; on the host prefer that env var, else the committed
    # host data dir, else sklearn's default cache.
    home = os.environ.get("SKLEARN_DATA_HOME")
    if home and Path(home).is_dir():
        return home
    cand = _PROJECT_ROOT / "vendor" / "data" / "sklearn"
    if cand.is_dir():
        return str(cand)
    return home


def load_dataset(name):
    """Load a dataset by name. Returns X, y, task_type."""
    if name == "breast_cancer":
        data = load_breast_cancer()
        return data.data, data.target, "classification"
    elif name == "diabetes":
        data = load_diabetes()
        return data.data, data.target, "regression"
    elif name == "california_housing":
        data = fetch_california_housing(data_home=_sklearn_data_home())
        return data.data, data.target, "regression"
    else:
        raise ValueError(f"Unknown dataset: {name}")


def normalize_features(X_train, X_test):
    """Standardize features to zero mean and unit variance."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def _split_normalized(env_name, seed):
    """Reproduce the exact original split + normalization for an env label."""
    dataset, task_type = DATASET_SPECS[env_name]
    X, y, _detected = load_dataset(dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed,
    )
    X_train, X_test = normalize_features(X_train, X_test)
    return X_train, y_train, X_test, y_test, task_type


def score_predictions(env_name, preds, y_test):
    """Compute the same final metric as before for the env's task type.

    Classification -> {"test_accuracy": accuracy_score(...)}.
    Regression     -> {"test_rmse": sqrt(mean_squared_error(...))}.
    """
    _dataset, task_type = DATASET_SPECS[env_name]
    if task_type == "classification":
        preds = np.asarray(preds).astype(int)
        return {"test_accuracy": float(accuracy_score(y_test, preds))}
    else:
        preds = np.asarray(preds, dtype=np.float64)
        return {"test_rmse": float(np.sqrt(mean_squared_error(y_test, preds)))}


# ---- helpers for the input pre-generator and the host-side scorer ----

def gen_input(env_name, seed=42):
    """Agent-visible input: (X_train, y_train, X_test) plus the fixed pipeline
    hyperparameters and the task type. y_test is withheld.

    The agent legitimately needs y_train to fit the weak learners; only the
    held-out test labels are withheld.
    """
    X_train, y_train, X_test, _y_test, task_type = _split_normalized(env_name, seed)
    return (
        X_train, y_train, X_test, task_type,
        N_ROUNDS, MAX_DEPTH, LEARNING_RATE,
    )


def truth(env_name, seed=42):
    """Held-out ground truth: y_test for the same split."""
    _Xtr, _ytr, _Xte, y_test, _task = _split_normalized(env_name, seed)
    return y_test


def opaque_label(label):
    """Host-only opaque token for a dataset label (the agent must not learn which
    dataset is evaluated). Used to name the pre-generated input files and as the
    identity the run scripts pass, so listing the workspace reveals only tokens;
    the real label stays host-side (the scorer uses the test-command label)."""
    import hashlib
    return hashlib.sha1(
        ("ml-ensemble-boosting::dataset-token::v1::" + str(label)).encode("utf-8")
    ).hexdigest()[:12]
