"""Custom clustering algorithm benchmark.

This script evaluates a clustering algorithm across multiple dataset types.
The agent should modify the EDITABLE section to implement a novel clustering
algorithm or distance metric that achieves high cluster quality.

Datasets (selected by $ENV):
  - blobs:          Isotropic Gaussian blobs (varying cluster sizes)
  - moons:          Two interleaving half-circles + noise
  - varied_density: Clusters with different densities and sizes
  - digits:         Real-world: sklearn Digits (8x8 images of handwritten digits)

Metrics: ARI (Adjusted Rand Index), NMI (Normalized Mutual Information),
         Silhouette Score
"""

import os
import sys
import warnings
import numpy as np
from sklearn.datasets import make_blobs, make_moons, load_digits
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
# FIXED -- do not modify below this line
# ================================================================


def generate_dataset(env_name, seed=42):
    """Generate or load the dataset for the given environment."""
    rng = np.random.RandomState(seed)

    if env_name == "blobs":
        X, y = make_blobs(
            n_samples=1500,
            centers=5,
            cluster_std=[0.8, 1.2, 0.5, 1.5, 1.0],
            random_state=seed,
        )
        n_clusters_true = 5
    elif env_name == "moons":
        X, y = make_moons(n_samples=1000, noise=0.08, random_state=seed)
        n_clusters_true = 2
    elif env_name == "varied_density":
        # Three clusters with very different densities
        X1, y1 = make_blobs(
            n_samples=500, centers=[[0, 0]], cluster_std=0.3, random_state=seed
        )
        X2, y2 = make_blobs(
            n_samples=300,
            centers=[[4, 4]],
            cluster_std=1.5,
            random_state=seed + 1,
        )
        X3, y3 = make_blobs(
            n_samples=200,
            centers=[[-3, 5]],
            cluster_std=0.6,
            random_state=seed + 2,
        )
        X = np.vstack([X1, X2, X3])
        y = np.concatenate([y1, y2 + 1, y3 + 2])
        n_clusters_true = 3
    elif env_name == "digits":
        digits = load_digits()
        X, y = digits.data, digits.target
        n_clusters_true = 10
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    return X, y, n_clusters_true


def evaluate_clustering(X, y_true, labels):
    """Compute clustering quality metrics."""
    metrics = {}
    metrics["ari"] = adjusted_rand_score(y_true, labels)
    metrics["nmi"] = normalized_mutual_info_score(y_true, labels)
    # Silhouette requires at least 2 clusters and fewer than n_samples
    n_labels = len(set(labels)) - (1 if -1 in labels else 0)
    if 2 <= n_labels < len(X):
        metrics["silhouette"] = silhouette_score(X, labels)
    else:
        metrics["silhouette"] = -1.0
    return metrics


def main():
    env = os.environ.get("ENV", "blobs")
    seed = int(os.environ.get("SEED", "42"))

    print(f"=== Clustering benchmark: {env} (seed={seed}) ===", flush=True)

    # Generate data
    X_raw, y_true, n_clusters_true = generate_dataset(env, seed=seed)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    print(f"Dataset: {env}, samples={X.shape[0]}, features={X.shape[1]}, "
          f"true_clusters={n_clusters_true}", flush=True)

    # Run custom clustering
    print("TRAIN_METRICS stage=fitting", flush=True)
    model = CustomClustering(n_clusters=n_clusters_true, random_state=seed)
    model.fit(X)
    labels = model.predict(X)
    print("TRAIN_METRICS stage=done", flush=True)

    # Evaluate
    metrics = evaluate_clustering(X, y_true, labels)

    for k, v in metrics.items():
        print(f"TRAIN_METRICS {k}={v:.6f}", flush=True)

    # Final metrics
    parts = " ".join(f"{k}={v:.6f}" for k, v in metrics.items())
    print(f"TEST_METRICS {parts}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
