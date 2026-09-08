"""Custom clustering algorithm benchmark.

This script evaluates a clustering algorithm across multiple dataset types.
The agent should modify the EDITABLE section to implement a novel clustering
algorithm or distance metric that achieves high cluster quality.

Several standard clustering datasets are used for evaluation (a mix of
synthetic and real, with differing shapes, densities, and cluster counts).
The specific datasets, their parameters, and which $ENV maps to which are
deliberately withheld from this file; your algorithm receives the standardized
feature matrix and the (metadata) number of clusters, and must generalize.

Metrics: ARI (Adjusted Rand Index), NMI (Normalized Mutual Information),
         Silhouette Score
"""

import os
import sys
import warnings
import numpy as np
# (dataset generators live in the host-only scoring module; eval datasets not named here)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.base import BaseEstimator, ClusterMixin

warnings.filterwarnings("ignore")

# ================================================================
# FIXED -- do not modify above this line
# ================================================================

# ================================================================
# EDITABLE -- agent modifies this section (lines 36 to 109)
# ================================================================


class CustomClustering(BaseEstimator, ClusterMixin):
    """Custom clustering algorithm.

    Must implement:
        fit(X) -> self          : fit the model to data X (n_samples, n_features)
        predict(X) -> labels    : return cluster labels for X (n_samples,)

    The algorithm should:
    - Automatically determine the number of clusters when n_clusters is None
    - Handle datasets with varying densities, non-convex shapes, and noise
    - Work well on both synthetic and real-world data

    Args:
        n_clusters: Number of clusters. If None, the algorithm should
                    determine this automatically.
        random_state: Random seed for reproducibility.
    """

    def __init__(self, n_clusters=None, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None

    def fit(self, X):
        """Fit the clustering model to data X.

        Args:
            X: array of shape (n_samples, n_features)

        Returns:
            self
        """
        # Default: simple K-Means fallback
        from sklearn.cluster import KMeans

        k = self.n_clusters if self.n_clusters is not None else 8
        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        km.fit(X)
        self.labels_ = km.labels_
        return self

    def predict(self, X):
        """Predict cluster labels for X.

        Args:
            X: array of shape (n_samples, n_features)

        Returns:
            labels: array of shape (n_samples,) with cluster assignments
        """
        # Default: refit (stateless fallback)
        self.fit(X)
        return self.labels_


# Placeholder for optional custom distance metric
def custom_distance(x, y):
    """Custom distance metric between two points.

    Args:
        x, y: 1-D arrays of shape (n_features,)

    Returns:
        distance: float >= 0
    """
    return np.sqrt(np.sum((x - y) ** 2))


# ================================================================
# ================================================================
# FIXED -- input loading + prediction emit (do not modify below this line)
# ================================================================
# The dataset generator (incl. identity), the true labels, and the metrics live
# in a host-only module the agent's process cannot import. This program loads
# the pre-generated standardized matrix, runs the clusterer, and emits the
# cluster labels; the host-side parser regenerates the truth and scores it.


def _cluster_inputs_dir():
    d = os.environ.get("CLUSTER_INPUTS_DIR")
    if d:
        return d
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cluster_inputs")


def _load_input(env_name, seed):
    import io as _io, base64 as _b64
    path = os.path.join(_cluster_inputs_dir(), f"{env_name}_seed{seed}.npz.b64")
    with open(path, "r") as f:
        raw = _b64.b64decode(f.read())
    d = np.load(_io.BytesIO(raw))
    return d["X"], int(d["n_clusters"])


def main():
    import base64 as _b64
    env = os.environ.get("ENV", "")
    if not env:
        raise SystemExit("ENV not set")
    seed = int(os.environ.get("SEED", "42"))
    print(f"=== Clustering benchmark: {env} (seed={seed}) ===", flush=True)

    X, n_clusters_true = _load_input(env, seed)
    print(f"Input: samples={X.shape[0]}, features={X.shape[1]}, "
          f"true_clusters={n_clusters_true}", flush=True)

    print("TRAIN_METRICS stage=fitting", flush=True)
    model = CustomClustering(n_clusters=n_clusters_true, random_state=seed)
    model.fit(X)
    labels = np.asarray(model.predict(X))
    print("TRAIN_METRICS stage=done", flush=True)

    payload = _b64.b64encode(np.ascontiguousarray(labels, dtype=np.int64).tobytes()).decode("ascii")
    print(f"CLUSTER_PRED env={env} seed={seed} n={labels.shape[0]} labels={payload}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
