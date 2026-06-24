"""Held-out scoring module for ml-clustering-algorithm (NOT agent-visible).

Lives outside every path bind-mounted into the agent container; imported only
by the host-side parser.py (native) / Harbor verifier. Holds the dataset
generator (incl. the dataset identity) and the ARI/NMI metrics, so the editable
clusterer can neither name the eval datasets nor regenerate the true labels.

Bodies are byte-identical to the originals, so honest results are reproduced
exactly.
"""

import numpy as np
from sklearn.datasets import make_blobs, make_moons, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


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
        X1, y1 = make_blobs(
            n_samples=500, centers=[[0, 0]], cluster_std=0.3, random_state=seed
        )
        X2, y2 = make_blobs(
            n_samples=300, centers=[[4, 4]], cluster_std=1.5, random_state=seed + 1
        )
        X3, y3 = make_blobs(
            n_samples=200, centers=[[-3, 5]], cluster_std=0.6, random_state=seed + 2
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
    """Compute clustering quality metrics (same as before)."""
    metrics = {}
    metrics["ari"] = adjusted_rand_score(y_true, labels)
    metrics["nmi"] = normalized_mutual_info_score(y_true, labels)
    n_labels = len(set(labels)) - (1 if -1 in labels else 0)
    if 2 <= n_labels < len(X):
        metrics["silhouette"] = silhouette_score(X, labels)
    else:
        metrics["silhouette"] = -1.0
    return metrics


# ---- helpers for the input pre-generator and the host-side scorer ----

def gen_input(env_name, seed=42):
    """Return the agent-visible input: the standardized feature matrix and the
    (metadata) true cluster count. The true cluster *assignments* are withheld."""
    X_raw, _y, n_clusters_true = generate_dataset(env_name, seed=seed)
    X = StandardScaler().fit_transform(X_raw)
    return X, int(n_clusters_true)


def truth(env_name, seed=42):
    """Return held-out ground truth for scoring: (X_standardized, y_true)."""
    X_raw, y_true, _n = generate_dataset(env_name, seed=seed)
    X = StandardScaler().fit_transform(X_raw)
    return X, y_true
