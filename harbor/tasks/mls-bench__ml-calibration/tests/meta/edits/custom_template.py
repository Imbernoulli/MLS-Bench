"""ML Calibration Benchmark.

Evaluate post-hoc probability calibration methods across different classifiers
and datasets.

FIXED: Classifier training, data loading, evaluation metrics, train/calibrate/test split.
EDITABLE: CalibrationMethod class (fit + predict_proba).

Usage:
    python scikit-learn/custom_calibration.py \
        --classifier rf --dataset mnist --seed 42
"""

import argparse
import math
import os
import warnings

import numpy as np
from scipy import optimize, interpolate, special
from sklearn.base import BaseEstimator
from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
)
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer, label_binarize
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# ============================================================================
# Calibration Method (EDITABLE)
# ============================================================================

# -- EDITABLE REGION START (lines 45-102) ------------------------------------
class CalibrationMethod(BaseEstimator):
    """Post-hoc probability calibration method.

    Given a trained classifier's uncalibrated probability outputs, learn a
    calibration mapping that produces well-calibrated probabilities.

    For binary classification, fit() receives probabilities for the positive
    class. For multiclass, it receives the full probability matrix.

    Interface:
        fit(probs, labels):
            probs: np.ndarray, shape (n_samples,) for binary or
                   (n_samples, n_classes) for multiclass.
                   Uncalibrated probability outputs from a classifier
                   on the calibration set.
            labels: np.ndarray, shape (n_samples,) integer class labels.

        predict_proba(probs) -> np.ndarray:
            probs: same format as fit().
            Returns calibrated probabilities, same shape as input.
            For binary: 1-D array of positive-class probabilities in [0, 1].
            For multiclass: 2-D array (n_samples, n_classes), rows sum to 1.

    Design considerations:
        - Parametric vs non-parametric calibration mappings
        - Monotonicity preservation (calibration should not reorder predictions)
        - Overfitting on small calibration sets
        - Multiclass extension strategy (per-class, matrix, or joint)
        - Binning vs continuous calibration functions
        - Regularization to prevent extreme probability outputs
    """

    def __init__(self):
        self.is_binary = None

    def fit(self, probs, labels):
        """Fit calibration mapping on held-out calibration data.

        Default: identity (no calibration).
        """
        if probs.ndim == 1:
            self.is_binary = True
        else:
            self.is_binary = False
        return self

    def predict_proba(self, probs):
        """Apply calibration mapping to produce calibrated probabilities.

        Default: return uncalibrated probabilities unchanged.
        """
        if self.is_binary:
            return np.clip(probs, 0, 1)
        else:
            # Ensure rows sum to 1
            probs = np.clip(probs, 1e-15, 1.0)
            probs = probs / probs.sum(axis=1, keepdims=True)
            return probs
# -- EDITABLE REGION END (lines 45-102) --------------------------------------


# ============================================================================
# Evaluation Metrics (FIXED)
# ============================================================================

def expected_calibration_error(probs, labels, n_bins=15):
    """Compute Expected Calibration Error (ECE).

    For binary: probs is 1-D (positive class probability).
    For multiclass: probs is 2-D, we compute ECE on the max-class probability
    and whether the argmax prediction is correct.

    Lower is better.
    """
    if probs.ndim == 2:
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
    else:
        confidences = np.where(probs >= 0.5, probs, 1 - probs)
        predictions = (probs >= 0.5).astype(int)
        accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        prop = in_bin.mean()
        if prop > 0:
            avg_conf = confidences[in_bin].mean()
            avg_acc = accuracies[in_bin].mean()
            ece += prop * abs(avg_acc - avg_conf)
    return ece


def compute_brier_score(probs, labels, n_classes):
    """Compute Brier score (multi-class generalization).

    Lower is better. Range [0, 2] for multiclass, [0, 1] for binary.
    """
    if n_classes == 2 and probs.ndim == 1:
        return brier_score_loss(labels, probs)
    else:
        lb = LabelBinarizer()
        lb.classes_ = np.arange(n_classes)
        labels_onehot = lb.transform(labels)
        if n_classes == 2:
            labels_onehot = np.column_stack([1 - labels_onehot, labels_onehot])
        if probs.ndim == 1:
            probs = np.column_stack([1 - probs, probs])
        return np.mean(np.sum((probs - labels_onehot) ** 2, axis=1))


def compute_nll(probs, labels, n_classes):
    """Compute negative log-likelihood (cross-entropy).

    Lower is better.
    """
    if probs.ndim == 1:
        probs_2d = np.column_stack([1 - probs, probs])
    else:
        probs_2d = probs
    probs_2d = np.clip(probs_2d, 1e-15, 1 - 1e-15)
    return log_loss(labels, probs_2d, labels=np.arange(n_classes))


# ============================================================================
# Data Loading (FIXED)
# ============================================================================

def load_dataset(name, seed=42):
    """Load and split dataset into train/calibrate/test.

    Split ratios: 60% train, 20% calibrate, 20% test.
    Returns: X_train, y_train, X_cal, y_cal, X_test, y_test, n_classes
    """
    if name == "mnist":
        data = fetch_openml("mnist_784", version=1, data_home=os.environ.get("SKLEARN_DATA_HOME", "/data/sklearn"),
                            parser="auto", as_frame=False)
        X, y = data["data"].astype(np.float32), data["target"].astype(int)
        # Subsample to 20000 for speed
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), 20000, replace=False)
        X, y = X[idx], y[idx]
        X = X / 255.0
        n_classes = 10
    elif name == "fashion_mnist":
        data = fetch_openml("Fashion-MNIST", version=1, data_home=os.environ.get("SKLEARN_DATA_HOME", "/data/sklearn"),
                            parser="auto", as_frame=False)
        X, y = data["data"].astype(np.float32), data["target"].astype(int)
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), 20000, replace=False)
        X, y = X[idx], y[idx]
        X = X / 255.0
        n_classes = 10
    elif name == "breast_cancer":
        data = load_breast_cancer()
        X, y = data["data"].astype(np.float32), data["target"].astype(int)
        # Standardize features
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        X = (X - mean) / std
        n_classes = 2
    elif name == "madelon":
        data = fetch_openml("madelon", version=1, data_home=os.environ.get("SKLEARN_DATA_HOME", "/data/sklearn"),
                            parser="auto", as_frame=False)
        X = data["data"].astype(np.float32)
        y_raw = data["target"]
        # madelon labels can be {1, 2} or {-1, 1}; normalize to {0, 1}
        unique_labels = np.unique(y_raw)
        label_map = {lab: i for i, lab in enumerate(sorted(unique_labels))}
        y = np.array([label_map[lab] for lab in y_raw])
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        X = (X - mean) / std
        n_classes = 2
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Split: train 60%, calibrate 20%, test 20%
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=seed, stratify=y_trainval
    )
    return X_train, y_train, X_cal, y_cal, X_test, y_test, n_classes


# ============================================================================
# Classifier Training (FIXED)
# ============================================================================

def build_classifier(name, seed=42):
    """Build an uncalibrated classifier."""
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=2,
            random_state=seed, n_jobs=-1,
        )
    elif name == "svm":
        return SVC(
            kernel="rbf", C=10.0, gamma="scale",
            probability=True, random_state=seed,
        )
    elif name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(256, 128), activation="relu",
            max_iter=200, early_stopping=True, validation_fraction=0.1,
            random_state=seed,
        )
    elif name == "gbm":
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=seed,
        )
    elif name == "lr":
        return LogisticRegression(
            C=1.0, max_iter=1000, random_state=seed,
        )
    else:
        raise ValueError(f"Unknown classifier: {name}")


# ============================================================================
# Main Pipeline (FIXED)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ML Calibration Benchmark")
    parser.add_argument("--classifier", type=str, required=True,
                        choices=["rf", "svm", "mlp", "gbm", "lr"])
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["mnist", "fashion_mnist", "breast_cancer", "madelon"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load data
    print(f"Loading dataset: {args.dataset}", flush=True)
    X_train, y_train, X_cal, y_cal, X_test, y_test, n_classes = load_dataset(
        args.dataset, seed=args.seed
    )
    print(f"  Train: {X_train.shape}, Cal: {X_cal.shape}, Test: {X_test.shape}, "
          f"Classes: {n_classes}", flush=True)

    # Train classifier
    print(f"Training classifier: {args.classifier}", flush=True)
    clf = build_classifier(args.classifier, seed=args.seed)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"TRAIN_METRICS: train_acc={train_acc:.4f} test_acc={test_acc:.4f}", flush=True)

    # Get uncalibrated probabilities
    if n_classes == 2:
        cal_probs_uncal = clf.predict_proba(X_cal)[:, 1]
        test_probs_uncal = clf.predict_proba(X_test)[:, 1]
    else:
        cal_probs_uncal = clf.predict_proba(X_cal)
        test_probs_uncal = clf.predict_proba(X_test)

    # Evaluate BEFORE calibration
    ece_before = expected_calibration_error(test_probs_uncal, y_test)
    brier_before = compute_brier_score(test_probs_uncal, y_test, n_classes)
    nll_before = compute_nll(test_probs_uncal, y_test, n_classes)
    print(f"TRAIN_METRICS: before_calibration ECE={ece_before:.6f} "
          f"Brier={brier_before:.6f} NLL={nll_before:.6f}", flush=True)

    # Fit calibration method on calibration set
    print("Fitting calibration method...", flush=True)
    calibrator = CalibrationMethod()
    calibrator.fit(cal_probs_uncal, y_cal)

    # Apply calibration to test set
    test_probs_cal = calibrator.predict_proba(test_probs_uncal)

    # Validate output shape and values
    assert test_probs_cal.shape == test_probs_uncal.shape, \
        f"Shape mismatch: {test_probs_cal.shape} vs {test_probs_uncal.shape}"
    if test_probs_cal.ndim == 2:
        row_sums = test_probs_cal.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-3), \
            f"Rows do not sum to 1: min={row_sums.min():.4f}, max={row_sums.max():.4f}"

    # Evaluate AFTER calibration
    ece_after = expected_calibration_error(test_probs_cal, y_test)
    brier_after = compute_brier_score(test_probs_cal, y_test, n_classes)
    nll_after = compute_nll(test_probs_cal, y_test, n_classes)

    print(f"TRAIN_METRICS: after_calibration ECE={ece_after:.6f} "
          f"Brier={brier_after:.6f} NLL={nll_after:.6f}", flush=True)

    # Final metrics (improvement = reduction in error)
    print(f"TEST_METRICS: ECE={ece_after:.6f} Brier={brier_after:.6f} "
          f"NLL={nll_after:.6f}", flush=True)


if __name__ == "__main__":
    main()
