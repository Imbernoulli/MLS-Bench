# MLS-Bench: ml-clustering-algorithm

# Clustering Algorithm Design

## Research Question
Design a novel clustering algorithm — and, if useful, an associated distance/affinity model — that improves cluster quality across diverse dataset geometries: convex blobs, non-convex shapes, and high-dimensional embeddings. The contribution is the *algorithm itself* (assignment rule, graph construction, density estimation, initialization, ensembling, ...), not dataset-specific tricks.

## Background
Clustering partitions unlabeled data into groups that reflect the underlying structure. No single classical method dominates across geometries.

Reference baselines:
- **K-Means** — Lloyd, 1957 / MacQueen, 1967. Iteratively assigns each point to the nearest of `K` centroids; assumes convex, isotropic clusters. Default initialization here: `k-means++`.
- **DBSCAN** — Ester, Kriegel, Sander, Xu, KDD 1996 ([paper](https://file.biolab.si/papers/1996-DBSCAN-KDD.pdf)). Density-based: a point is a *core* point if it has ≥ `min_samples` neighbors within radius `eps`; clusters are connected components of core points; non-core neighbors are border points; the rest are noise.
- **HDBSCAN** — Campello, Moulavi, Sander, PAKDD 2013 ([paper](https://link.springer.com/chapter/10.1007/978-3-642-37456-2_14)). Hierarchical density-based clustering with mutual reachability distances and a stability-based flat extraction; only `min_cluster_size` (and optional `min_samples`) needed.

## Implementation Contract
Modify `CustomClustering` (and optionally the `custom_distance` helper) in `scikit-learn/custom_clustering.py`:

```python
class CustomClustering(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=None, random_state=42):
        ...
    def fit(self, X):       # X: (n_samples, n_features) -> self (sets self.labels_)
        ...
    def predict(self, X):   # X: (n_samples, n_features) -> int cluster labels
        ...
```

Available imports (already in the FIXED section): `numpy`, `sklearn.base.BaseEstimator`, `sklearn.base.ClusterMixin`, `sklearn.preprocessing.StandardScaler`, `sklearn.metrics.*`. You may import any module from `scikit-learn`, `numpy`, or `scipy`.

## Fixed Pipeline & Evaluation
Datasets:
- **blobs** — 5 isotropic Gaussian clusters.
- **moons** — 2 interleaving half-circles (non-convex).
- **digits** — `sklearn.datasets.load_digits()`, 10 classes, 64 features (high-dimensional real data).

Metrics (higher is better):
- **ARI** — Adjusted Rand Index (matches predicted clusters to ground-truth labels).
- **NMI** — Normalized Mutual Information.
- **Silhouette** — intrinsic compactness/separation of clusters.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scikit-learn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `scikit-learn/custom_clustering.py`
- editable lines **36–109**




## Readable Context


### `scikit-learn/custom_clustering.py`  [EDITABLE — lines 36–109 only]

```python
     1: """Custom clustering algorithm benchmark.
     2: 
     3: This script evaluates a clustering algorithm across multiple dataset types.
     4: The agent should modify the EDITABLE section to implement a novel clustering
     5: algorithm or distance metric that achieves high cluster quality.
     6: 
     7: Datasets (selected by $ENV):
     8:   - blobs:          Isotropic Gaussian blobs (varying cluster sizes)
     9:   - moons:          Two interleaving half-circles + noise
    10:   - varied_density: Clusters with different densities and sizes
    11:   - digits:         Real-world: sklearn Digits (8x8 images of handwritten digits)
    12: 
    13: Metrics: ARI (Adjusted Rand Index), NMI (Normalized Mutual Information),
    14:          Silhouette Score
    15: """
    16: 
    17: import os
    18: import sys
    19: import warnings
    20: import numpy as np
    21: from sklearn.datasets import make_blobs, make_moons, load_digits
    22: from sklearn.preprocessing import StandardScaler
    23: from sklearn.metrics import (
    24:     adjusted_rand_score,
    25:     normalized_mutual_info_score,
    26:     silhouette_score,
    27: )
    28: from sklearn.base import BaseEstimator, ClusterMixin
    29: 
    30: warnings.filterwarnings("ignore")
    31: 
    32: # ================================================================
    33: # FIXED -- do not modify above this line
    34: # ================================================================
    35: 
    36: # ================================================================
    37: # EDITABLE -- agent modifies this section (lines 36 to 109)
    38: # ================================================================
    39: 
    40: 
    41: class CustomClustering(BaseEstimator, ClusterMixin):
    42:     """Custom clustering algorithm.
    43: 
    44:     Must implement:
    45:         fit(X) -> self          : fit the model to data X (n_samples, n_features)
    46:         predict(X) -> labels    : return cluster labels for X (n_samples,)
    47: 
    48:     The algorithm should:
    49:     - Automatically determine the number of clusters when n_clusters is None
    50:     - Handle datasets with varying densities, non-convex shapes, and noise
    51:     - Work well on both synthetic and real-world data
    52: 
    53:     Args:
    54:         n_clusters: Number of clusters. If None, the algorithm should
    55:                     determine this automatically.
    56:         random_state: Random seed for reproducibility.
    57:     """
    58: 
    59:     def __init__(self, n_clusters=None, random_state=42):
    60:         self.n_clusters = n_clusters
    61:         self.random_state = random_state
    62:         self.labels_ = None
    63: 
    64:     def fit(self, X):
    65:         """Fit the clustering model to data X.
    66: 
    67:         Args:
    68:             X: array of shape (n_samples, n_features)
    69: 
    70:         Returns:
    71:             self
    72:         """
    73:         # Default: simple K-Means fallback
    74:         from sklearn.cluster import KMeans
    75: 
    76:         k = self.n_clusters if self.n_clusters is not None else 8
    77:         km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
    78:         km.fit(X)
    79:         self.labels_ = km.labels_
    80:         return self
    81: 
    82:     def predict(self, X):
    83:         """Predict cluster labels for X.
    84: 
    85:         Args:
    86:             X: array of shape (n_samples, n_features)
    87: 
    88:         Returns:
    89:             labels: array of shape (n_samples,) with cluster assignments
    90:         """
    91:         # Default: refit (stateless fallback)
    92:         self.fit(X)
    93:         return self.labels_
    94: 
    95: 
    96: # Placeholder for optional custom distance metric
    97: def custom_distance(x, y):
    98:     """Custom distance metric between two points.
    99: 
   100:     Args:
   101:         x, y: 1-D arrays of shape (n_features,)
   102: 
   103:     Returns:
   104:         distance: float >= 0
   105:     """
   106:     return np.sqrt(np.sum((x - y) ** 2))
   107: 
   108: 
   109: # ================================================================
   110: # FIXED -- do not modify below this line
   111: # ================================================================
   112: 
   113: 
   114: def generate_dataset(env_name, seed=42):
   115:     """Generate or load the dataset for the given environment."""
   116:     rng = np.random.RandomState(seed)
   117: 
   118:     if env_name == "blobs":
   119:         X, y = make_blobs(
   120:             n_samples=1500,
   121:             centers=5,
   122:             cluster_std=[0.8, 1.2, 0.5, 1.5, 1.0],
   123:             random_state=seed,
   124:         )
   125:         n_clusters_true = 5
   126:     elif env_name == "moons":
   127:         X, y = make_moons(n_samples=1000, noise=0.08, random_state=seed)
   128:         n_clusters_true = 2
   129:     elif env_name == "varied_density":
   130:         # Three clusters with very different densities
   131:         X1, y1 = make_blobs(
   132:             n_samples=500, centers=[[0, 0]], cluster_std=0.3, random_state=seed
   133:         )
   134:         X2, y2 = make_blobs(
   135:             n_samples=300,
   136:             centers=[[4, 4]],
   137:             cluster_std=1.5,
   138:             random_state=seed + 1,
   139:         )
   140:         X3, y3 = make_blobs(
   141:             n_samples=200,
   142:             centers=[[-3, 5]],
   143:             cluster_std=0.6,
   144:             random_state=seed + 2,
   145:         )
   146:         X = np.vstack([X1, X2, X3])
   147:         y = np.concatenate([y1, y2 + 1, y3 + 2])
   148:         n_clusters_true = 3
   149:     elif env_name == "digits":
   150:         digits = load_digits()
   151:         X, y = digits.data, digits.target
   152:         n_clusters_true = 10
   153:     else:
   154:         raise ValueError(f"Unknown environment: {env_name}")
   155: 
   156:     return X, y, n_clusters_true
   157: 
   158: 
   159: def evaluate_clustering(X, y_true, labels):
   160:     """Compute clustering quality metrics."""
   161:     metrics = {}
   162:     metrics["ari"] = adjusted_rand_score(y_true, labels)
   163:     metrics["nmi"] = normalized_mutual_info_score(y_true, labels)
   164:     # Silhouette requires at least 2 clusters and fewer than n_samples
   165:     n_labels = len(set(labels)) - (1 if -1 in labels else 0)
   166:     if 2 <= n_labels < len(X):
   167:         metrics["silhouette"] = silhouette_score(X, labels)
   168:     else:
   169:         metrics["silhouette"] = -1.0
   170:     return metrics
   171: 
   172: 
   173: def main():
   174:     env = os.environ.get("ENV", "blobs")
   175:     seed = int(os.environ.get("SEED", "42"))
   176: 
   177:     print(f"=== Clustering benchmark: {env} (seed={seed}) ===", flush=True)
   178: 
   179:     # Generate data
   180:     X_raw, y_true, n_clusters_true = generate_dataset(env, seed=seed)
   181: 
   182:     # Standardize features
   183:     scaler = StandardScaler()
   184:     X = scaler.fit_transform(X_raw)
   185: 
   186:     print(f"Dataset: {env}, samples={X.shape[0]}, features={X.shape[1]}, "
   187:           f"true_clusters={n_clusters_true}", flush=True)
   188: 
   189:     # Run custom clustering
   190:     print("TRAIN_METRICS stage=fitting", flush=True)
   191:     model = CustomClustering(n_clusters=n_clusters_true, random_state=seed)
   192:     model.fit(X)
   193:     labels = model.predict(X)
   194:     print("TRAIN_METRICS stage=done", flush=True)
   195: 
   196:     # Evaluate
   197:     metrics = evaluate_clustering(X, y_true, labels)
   198: 
   199:     for k, v in metrics.items():
   200:         print(f"TRAIN_METRICS {k}={v:.6f}", flush=True)
   201: 
   202:     # Final metrics
   203:     parts = " ".join(f"{k}={v:.6f}" for k, v in metrics.items())
   204:     print(f"TEST_METRICS {parts}", flush=True)
   205: 
   206:     print("Done.", flush=True)
   207: 
   208: 
   209: if __name__ == "__main__":
   210:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `kmeans` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_clustering.py`:

```python
Lines 36–64:
    33: # FIXED -- do not modify above this line
    34: # ================================================================
    35: 
    36: 
    37: class CustomClustering(BaseEstimator, ClusterMixin):
    38:     """K-Means clustering (Lloyd's algorithm)."""
    39: 
    40:     def __init__(self, n_clusters=None, random_state=42):
    41:         self.n_clusters = n_clusters
    42:         self.random_state = random_state
    43:         self.labels_ = None
    44:         self._model = None
    45: 
    46:     def fit(self, X):
    47:         from sklearn.cluster import KMeans
    48: 
    49:         k = self.n_clusters if self.n_clusters is not None else 8
    50:         self._model = KMeans(
    51:             n_clusters=k, random_state=self.random_state, n_init=10, max_iter=300
    52:         )
    53:         self._model.fit(X)
    54:         self.labels_ = self._model.labels_
    55:         return self
    56: 
    57:     def predict(self, X):
    58:         if self._model is None:
    59:             self.fit(X)
    60:         return self._model.predict(X)
    61: 
    62: 
    63: def custom_distance(x, y):
    64:     return np.sqrt(np.sum((x - y) ** 2))
    65: # FIXED -- do not modify below this line
    66: # ================================================================
    67: 
```

### `dbscan` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_clustering.py`:

```python
Lines 36–102:
    33: # FIXED -- do not modify above this line
    34: # ================================================================
    35: 
    36: 
    37: class CustomClustering(BaseEstimator, ClusterMixin):
    38:     """DBSCAN density-based clustering.
    39: 
    40:     Uses the sklearn demo (plot_dbscan.html) parameters as a strong
    41:     default: eps=0.3, min_samples=10 on StandardScaled 2D data. For
    42:     higher-dimensional data we fall back to the knee of the k-distance
    43:     graph (Ester et al. 1996) with proper Kneedle-style detection.
    44:     """
    45: 
    46:     def __init__(self, n_clusters=None, random_state=42):
    47:         self.n_clusters = n_clusters
    48:         self.random_state = random_state
    49:         self.labels_ = None
    50: 
    51:     def fit(self, X):
    52:         from sklearn.cluster import DBSCAN
    53:         from sklearn.neighbors import NearestNeighbors
    54: 
    55:         n_features = X.shape[1]
    56: 
    57:         if n_features <= 3:
    58:             # StandardScaled low-D data: sklearn's DBSCAN demo uses
    59:             # eps=0.3, min_samples=10 for blobs (cluster_std=0.4).
    60:             # Our task's varied-density blobs (cluster_std up to 1.5)
    61:             # merge at eps=0.3; grid search on the generator's output
    62:             # shows eps=0.22 maximizes ARI. See plot_dbscan.html.
    63:             eps = 0.22
    64:             min_samples = 10
    65:         else:
    66:             # High-D fallback: knee of k-distance graph.
    67:             min_samples = max(4, min(2 * n_features, 10))
    68:             k = min(min_samples, X.shape[0] - 1)
    69:             nn = NearestNeighbors(n_neighbors=k + 1)
    70:             nn.fit(X)
    71:             distances, _ = nn.kneighbors(X)
    72:             kth = np.sort(distances[:, -1])
    73:             # Kneedle: point of maximum distance from the chord between
    74:             # the first and last points of the sorted curve.
    75:             n = len(kth)
    76:             if n >= 3:
    77:                 xs = np.arange(n, dtype=float)
    78:                 ys = kth
    79:                 x1, x2 = xs[0], xs[-1]
    80:                 y1, y2 = ys[0], ys[-1]
    81:                 denom = np.hypot(x2 - x1, y2 - y1) + 1e-12
    82:                 dist_to_chord = np.abs(
    83:                     (y2 - y1) * xs - (x2 - x1) * ys + x2 * y1 - y2 * x1
    84:                 ) / denom
    85:                 idx = int(np.argmax(dist_to_chord))
    86:                 eps = float(kth[idx])
    87:             else:
    88:                 eps = float(kth[-1])
    89: 
    90:         self._model = DBSCAN(eps=eps, min_samples=min_samples)
    91:         self._model.fit(X)
    92:         self.labels_ = self._model.labels_
    93:         return self
    94: 
    95:     def predict(self, X):
    96:         if self.labels_ is None:
    97:             self.fit(X)
    98:         return self.labels_
    99: 
   100: 
   101: def custom_distance(x, y):
   102:     return np.sqrt(np.sum((x - y) ** 2))
   103: # FIXED -- do not modify below this line
   104: # ================================================================
   105: 
```

### `hdbscan` baseline — editable region  [READ-ONLY — reference implementation]

In `scikit-learn/custom_clustering.py`:

```python
Lines 36–77:
    33: # FIXED -- do not modify above this line
    34: # ================================================================
    35: 
    36: 
    37: class CustomClustering(BaseEstimator, ClusterMixin):
    38:     """HDBSCAN — hierarchical density-based clustering (Campello et al., 2013)."""
    39: 
    40:     def __init__(self, n_clusters=None, random_state=42):
    41:         self.n_clusters = n_clusters
    42:         self.random_state = random_state
    43:         self.labels_ = None
    44: 
    45:     def fit(self, X):
    46:         from sklearn.cluster import HDBSCAN
    47: 
    48:         # HDBSCAN automatically determines the number of clusters.
    49:         # min_cluster_size controls granularity.
    50:         min_cs = max(5, X.shape[0] // 50)
    51:         self._model = HDBSCAN(
    52:             min_cluster_size=min_cs,
    53:             min_samples=5,
    54:             cluster_selection_method="eom",
    55:         )
    56:         self._model.fit(X)
    57:         self.labels_ = self._model.labels_
    58: 
    59:         # If HDBSCAN assigns everything to noise (-1), fall back to
    60:         # labeling all points as cluster 0 to avoid degenerate metrics.
    61:         if len(set(self.labels_)) <= 1:
    62:             from sklearn.cluster import KMeans
    63:             k = self.n_clusters if self.n_clusters is not None else 8
    64:             km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
    65:             km.fit(X)
    66:             self.labels_ = km.labels_
    67: 
    68:         return self
    69: 
    70:     def predict(self, X):
    71:         if self.labels_ is None:
    72:             self.fit(X)
    73:         return self.labels_
    74: 
    75: 
    76: def custom_distance(x, y):
    77:     return np.sqrt(np.sum((x - y) ** 2))
    78: # FIXED -- do not modify below this line
    79: # ================================================================
    80: 
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
