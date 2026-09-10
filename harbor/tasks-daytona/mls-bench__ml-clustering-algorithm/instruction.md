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

## Fixed Pipeline
The data pipeline and evaluation harness are fixed by the harness and not editable. Cluster quality is measured by ARI (Adjusted Rand Index), NMI (Normalized Mutual Information), and Silhouette Score.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scikit-learn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits that change code outside these ranges — or creating new files, or
deleting whole files — will cause your submission to be invalid.

The line numbers mark an editable **region**, not a fixed line-count budget: you
may add or remove lines inside it. Only code outside the editable ranges must
stay unchanged.

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
     7: Several standard clustering datasets are used for evaluation (a mix of
     8: synthetic and real, with differing shapes, densities, and cluster counts).
     9: The specific datasets, their parameters, and which $ENV maps to which are
    10: deliberately withheld from this file; your algorithm receives the standardized
    11: feature matrix and the (metadata) number of clusters, and must generalize.
    12: 
    13: Metrics: ARI (Adjusted Rand Index), NMI (Normalized Mutual Information),
    14:          Silhouette Score
    15: """
    16: 
    17: import os
    18: import sys
    19: import warnings
    20: import numpy as np
    21: # (dataset generators live in the host-only scoring module; eval datasets not named here)
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
   110: # ================================================================
   111: # FIXED -- input loading + prediction emit (do not modify below this line)
   112: # ================================================================
   113: # The dataset generator (incl. identity), the true labels, and the metrics live
   114: # in a host-only module the agent's process cannot import. This program loads
   115: # the pre-generated standardized matrix, runs the clusterer, and emits the
   116: # cluster labels; the host-side parser regenerates the truth and scores it.
   117: 
   118: 
   119: def _cluster_inputs_dir():
   120:     d = os.environ.get("CLUSTER_INPUTS_DIR")
   121:     if d:
   122:         return d
   123:     return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cluster_inputs")
   124: 
   125: 
   126: def _load_input(env_name, seed):
   127:     import io as _io, base64 as _b64
   128:     path = os.path.join(_cluster_inputs_dir(), f"{env_name}_seed{seed}.npz.b64")
   129:     with open(path, "r") as f:
   130:         raw = _b64.b64decode(f.read())
   131:     d = np.load(_io.BytesIO(raw))
   132:     return d["X"], int(d["n_clusters"])
   133: 
   134: 
   135: def main():
   136:     import base64 as _b64
   137:     env = os.environ.get("ENV", "")
   138:     if not env:
   139:         raise SystemExit("ENV not set")
   140:     seed = int(os.environ.get("SEED", "42"))
   141:     print(f"=== Clustering benchmark: {env} (seed={seed}) ===", flush=True)
   142: 
   143:     X, n_clusters_true = _load_input(env, seed)
   144:     print(f"Input: samples={X.shape[0]}, features={X.shape[1]}, "
   145:           f"true_clusters={n_clusters_true}", flush=True)
   146: 
   147:     print("TRAIN_METRICS stage=fitting", flush=True)
   148:     model = CustomClustering(n_clusters=n_clusters_true, random_state=seed)
   149:     model.fit(X)
   150:     labels = np.asarray(model.predict(X))
   151:     print("TRAIN_METRICS stage=done", flush=True)
   152: 
   153:     payload = _b64.b64encode(np.ascontiguousarray(labels, dtype=np.int64).tobytes()).decode("ascii")
   154:     print(f"CLUSTER_PRED env={env} seed={seed} n={labels.shape[0]} labels={payload}", flush=True)
   155:     print("Done.", flush=True)
   156: 
   157: 
   158: if __name__ == "__main__":
   159:     main()
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
    65: # ================================================================
    66: # FIXED -- input loading + prediction emit (do not modify below this line)
    67: # ================================================================
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
   103: # ================================================================
   104: # FIXED -- input loading + prediction emit (do not modify below this line)
   105: # ================================================================
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
    78: # ================================================================
    79: # FIXED -- input loading + prediction emit (do not modify below this line)
    80: # ================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

## Time Budget

You have **5 hours** of wall-clock time before submission, covering
everything you do here: reading the code, editing it, and any trial runs
you launch.

Good luck.
