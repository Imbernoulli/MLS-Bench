"""Unsupervised Anomaly Detection Benchmark for MLS-Bench.

EDITABLE: CustomAnomalyDetector class -- the agent's anomaly detection algorithm.
FIXED: input loading + prediction emit. The dataset identity, the train/test
split, the test labels, and the metrics live in a host-only module the agent's
process cannot import; this program loads a pre-generated standardized
(train, test) pair, fits the detector unsupervised on the train split, and emits
the test anomaly scores. The host-side parser regenerates the labels and scores
AUROC + F1. Inputs are pre-standardized, exactly as before.
"""

import os
import io
import base64
import warnings

import numpy as np
from sklearn.base import BaseEstimator

warnings.filterwarnings("ignore")
SEED = int(os.environ.get("SEED", "42"))
np.random.seed(SEED)


# =====================================================================
# EDITABLE: Custom Anomaly Detector
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
# FIXED: input loading + prediction emit (do not modify below this line)
# =====================================================================
def _inputs_dir():
    d = os.environ.get("ANOMALY_INPUTS_DIR")
    if d:
        return d
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_anomaly_inputs")


def _load_input(env_name, seed):
    path = os.path.join(_inputs_dir(), f"{env_name}_seed{seed}.npz.b64")
    with open(path, "r") as f:
        raw = base64.b64decode(f.read())
    d = np.load(io.BytesIO(raw))
    return d["X_train"], d["X_test"]


def main():
    env = os.environ.get("ENV", "")
    if not env:
        raise SystemExit("ENV not set")
    seed = SEED
    print(f"=== Anomaly detection benchmark: {env} (seed={seed}) ===", flush=True)
    X_train, X_test = _load_input(env, seed)
    print(f"Input: train={X_train.shape}, test={X_test.shape}", flush=True)
    detector = CustomAnomalyDetector()
    detector.fit(X_train)
    scores = np.asarray(detector.decision_function(X_test), dtype=np.float64).ravel()
    payload = base64.b64encode(np.ascontiguousarray(scores, dtype=np.float64).tobytes()).decode("ascii")
    print(f"ANOMALY_PRED env={env} seed={seed} n={scores.shape[0]} scores={payload}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
