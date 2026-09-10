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
