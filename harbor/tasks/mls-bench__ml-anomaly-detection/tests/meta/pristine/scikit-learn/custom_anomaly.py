"""Unsupervised Anomaly Detection Benchmark for MLS-Bench.

FIXED: Data loading, evaluation pipeline, metrics computation.
EDITABLE: CustomAnomalyDetector class — the agent's anomaly detection algorithm.

Usage:
    ENV=cardio SEED=42 OUTPUT_DIR=./output python custom_anomaly.py
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.base import BaseEstimator


# =====================================================================
# FIXED: Configuration
# =====================================================================
SEED = int(os.environ.get("SEED", "42"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
DATASET_NAME = os.environ.get("ENV", "cardio")

DATA_DIR = os.environ.get("DATA_ROOT", "/data") + "/adbench"

# Dataset file mapping
DATASET_FILES = {
    "cardio": "6_cardio.npz",
    "thyroid": "38_thyroid.npz",
    "satellite": "30_satellite.npz",
    "shuttle": "32_shuttle.npz",
}

TRAIN_RATIO = 0.6  # Task-local 60/40 stratified train/test split.

warnings.filterwarnings("ignore")
np.random.seed(SEED)


# =====================================================================
# FIXED: Data loading
# =====================================================================
def load_dataset(name: str):
    """Load an anomaly detection dataset.

    Returns:
        X: feature matrix of shape (n_samples, n_features), float64
        y: binary labels of shape (n_samples,), 0=normal 1=anomaly
    """
    filename = DATASET_FILES[name]
    filepath = os.path.join(DATA_DIR, filename)
    data = np.load(filepath, allow_pickle=True)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.int32).ravel()
    # Ensure binary: 0=normal, 1=anomaly
    y = (y > 0).astype(np.int32)
    return X, y


# =====================================================================
# FIXED: Evaluation utilities
# =====================================================================
def evaluate_detector(detector, X_train, X_test, y_test):
    """Fit detector on training data and evaluate on test data.

    Args:
        detector: an object with fit(X) and decision_function(X) methods.
                  fit(X) trains on UNLABELED data (no y).
                  decision_function(X) returns anomaly scores (higher = more anomalous).
        X_train: training features (n_train, n_features)
        X_test: test features (n_test, n_features)
        y_test: test labels (n_test,), 0=normal, 1=anomaly

    Returns:
        dict with 'auroc' and 'f1' metrics
    """
    # Fit on training data (unsupervised — no labels)
    detector.fit(X_train)

    # Get anomaly scores on test data
    scores = detector.decision_function(X_test)

    # AUROC
    try:
        auroc = roc_auc_score(y_test, scores)
    except ValueError:
        auroc = 0.5  # fallback if only one class present

    # F1 at optimal threshold (using test set threshold for fair comparison)
    # Threshold at the contamination ratio percentile
    contamination = y_test.mean()
    if contamination > 0 and contamination < 1:
        threshold = np.percentile(scores, 100 * (1 - contamination))
        y_pred = (scores >= threshold).astype(int)
    else:
        y_pred = np.zeros_like(y_test)

    f1 = f1_score(y_test, y_pred, zero_division=0.0)

    return {"auroc": auroc, "f1": f1}


def run_evaluation(detector_cls, X, y, seed):
    """Run evaluation with a 60/40 stratified train/test split.

    This task uses a fixed 60/40 stratified split. It is inspired by
    ADBench-style held-out evaluation, but is not the ADBench 70/30 protocol.

    Args:
        detector_cls: callable that returns a fresh detector instance
        X: full feature matrix
        y: full label vector
        seed: random seed

    Returns:
        dict with auroc and f1 metrics
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1.0 - TRAIN_RATIO, stratify=y, random_state=seed,
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create fresh detector and evaluate
    detector = detector_cls()

    try:
        metrics = evaluate_detector(detector, X_train_scaled, X_test_scaled, y_test)
        print(
            f"TRAIN_METRICS split=60/40 "
            f"auroc={metrics['auroc']:.4f} f1={metrics['f1']:.4f}",
            flush=True,
        )
    except Exception as e:
        print(f"TRAIN_METRICS split=60/40 error={str(e)}", flush=True)
        metrics = {"auroc": 0.5, "f1": 0.0}

    return {
        "auroc_mean": float(metrics["auroc"]),
        "auroc_std": 0.0,
        "f1_mean": float(metrics["f1"]),
        "f1_std": 0.0,
    }


# =====================================================================
# EDITABLE: Custom Anomaly Detector (lines 160-212)
# =====================================================================
class CustomAnomalyDetector:
    """Custom unsupervised anomaly detection algorithm.

    You MUST implement:
        - __init__(self): initialize any hyperparameters and internal state
        - fit(self, X): train the detector on unlabeled data X (n_samples, n_features).
                        This is UNSUPERVISED — you do not receive labels.
        - decision_function(self, X): return anomaly scores for X.
                        Shape: (n_samples,). Higher scores = more anomalous.

    Available libraries (pre-installed):
        - numpy, scipy, scikit-learn (StandardScaler, PCA, KernelDensity, etc.)
        - pyod (IForest, LOF, OCSVM, ECOD, COPOD, KNN, HBOS, PCA, LODA, etc.)

    The detector will be evaluated on tabular anomaly detection benchmarks via
    a 60/40 stratified train/test split, measuring AUROC and F1.

    Design considerations:
        - Anomalies are rare (typically 2-30% of data)
        - Feature dimensions vary (6 to 36 features)
        - Dataset sizes vary (1,800 to 49,000 samples)
        - Data is pre-standardized before being passed to fit/decision_function
        - Your algorithm should work WITHOUT labels (unsupervised)
        - Consider: density estimation, distance-based, projection-based,
          ensemble methods, or hybrid approaches
    """

    def __init__(self):
        """Initialize the anomaly detector."""
        # Default: simple Isolation Forest wrapper
        from pyod.models.iforest import IForest

        self.model = IForest(random_state=SEED)

    def fit(self, X):
        """Fit the detector on unlabeled training data.

        Args:
            X: numpy array of shape (n_samples, n_features), standardized
        """
        self.model.fit(X)
        return self

    def decision_function(self, X):
        """Compute anomaly scores for input data.

        Args:
            X: numpy array of shape (n_samples, n_features), standardized

        Returns:
            scores: numpy array of shape (n_samples,), higher = more anomalous
        """
        return self.model.decision_function(X)


# =====================================================================
# FIXED: Main evaluation script
# =====================================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Dataset: {DATASET_NAME}, Seed: {SEED}", flush=True)

    # Load data
    X, y = load_dataset(DATASET_NAME)
    print(
        f"Loaded {DATASET_NAME}: {X.shape[0]} samples, {X.shape[1]} features, "
        f"{y.mean()*100:.1f}% anomalies",
        flush=True,
    )

    # Run evaluation
    start_time = time.time()
    results = run_evaluation(
        detector_cls=CustomAnomalyDetector,
        X=X,
        y=y,
        seed=SEED,
    )
    elapsed = time.time() - start_time

    print(f"\nResults on {DATASET_NAME} (seed={SEED}):", flush=True)
    print(
        f"  AUROC: {results['auroc_mean']:.4f} +/- {results['auroc_std']:.4f}",
        flush=True,
    )
    print(
        f"  F1:    {results['f1_mean']:.4f} +/- {results['f1_std']:.4f}",
        flush=True,
    )
    print(f"  Time:  {elapsed:.1f}s", flush=True)

    # Output final metrics for parser
    print(
        f"TEST_METRICS auroc={results['auroc_mean']:.6f} f1={results['f1_mean']:.6f}",
        flush=True,
    )
