# MLS-Bench: ml-clustering-algorithm

# Clustering Algorithm Design

## Research Question
Design a novel clustering algorithm — and, if useful, an associated distance/affinity model — that improves cluster quality across diverse dataset geometries: convex blobs, non-convex shapes, and high-dimensional embeddings. The contribution is the *algorithm itself* (assignment rule, graph construction, density estimation, initialization, ensembling, ...), not dataset-specific tricks.

## Background
Clustering partitions unlabeled data into groups that reflect the underlying structure. No single classical method dominates across geometries.

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

Available imports (already in the FIXED section): `numpy`, `sklearn.base.BaseEstimator`, `sklearn.base.ClusterMixin`, `sklearn.preprocessing.StandardScaler`. You may import any module from `scikit-learn`, `numpy`, or `scipy`.

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/scikit-learn/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will be rejected.

- `scikit-learn/custom_clustering.py`
- editable lines **36–109**

## Readable Context

### `scikit-learn/custom_clustering.py`  [EDITABLE — lines 36–109 only]

```python
     1: """Custom clustering algorithm scaffold."""
     2:
     3: import os
     4: import sys
     5: import warnings
     6: import numpy as np
     7: from sklearn.preprocessing import StandardScaler
     8: from sklearn.base import BaseEstimator, ClusterMixin
     9:
    10: warnings.filterwarnings("ignore")
    11:
    12: # ================================================================
    13: # FIXED -- do not modify above this line
    14: # ================================================================
    15:
    16: # ================================================================
    17: # EDITABLE -- agent modifies this section (lines 36 to 109)
    18: # ================================================================
    19:
    20:
    36:
    37: class CustomClustering(BaseEstimator, ClusterMixin):
    38:     """Custom clustering algorithm.
    39:
    40:     Must implement:
    41:         fit(X) -> self          : fit the model to data X (n_samples, n_features)
    42:         predict(X) -> labels    : return cluster labels for X (n_samples,)
    43:
    44:     The algorithm should:
    45:     - Automatically determine the number of clusters when n_clusters is None
    46:     - Handle datasets with varying densities, non-convex shapes, and noise
    47:     - Work well on both synthetic and real-world data
    48:
    49:     Args:
    50:         n_clusters: Number of clusters. If None, the algorithm should
    51:                     determine this automatically.
    52:         random_state: Random seed for reproducibility.
    53:     """
    54:
    55:     def __init__(self, n_clusters=None, random_state=42):
    56:         self.n_clusters = n_clusters
    57:         self.random_state = random_state
    58:         self.labels_ = None
    59:
    60:     def fit(self, X):
    61:         from sklearn.cluster import KMeans
    62:         k = self.n_clusters if self.n_clusters is not None else 8
    63:         km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
    64:         km.fit(X)
    65:         self.labels_ = km.labels_
    66:         return self
    67:
    68:     def predict(self, X):
    69:         self.fit(X)
    70:         return self.labels_
    71:
    72:
    73: # Placeholder for optional custom distance metric
    74: def custom_distance(x, y):
    75:     return np.sqrt(np.sum((x - y) ** 2))
   109: # ================================================================
```

## Tips

- Keep the function/class signatures of the editable regions identical.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.

Good luck.
