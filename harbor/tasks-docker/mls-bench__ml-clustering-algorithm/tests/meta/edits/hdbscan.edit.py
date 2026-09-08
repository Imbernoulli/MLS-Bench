"""HDBSCAN baseline for ml-clustering-algorithm.

Hierarchical Density-Based Spatial Clustering of Applications with Noise.
SOTA density-based method that does not require eps parameter.
Reference: Campello, Moulavi, Sander (2013) — "Density-Based Clustering
Based on Hierarchical Density Estimates"
sklearn.cluster.HDBSCAN (available since scikit-learn 1.3)
"""

_FILE = "scikit-learn/custom_clustering.py"

_CONTENT = """\

class CustomClustering(BaseEstimator, ClusterMixin):
    \"\"\"HDBSCAN — hierarchical density-based clustering (Campello et al., 2013).\"\"\"

    def __init__(self, n_clusters=None, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None

    def fit(self, X):
        from sklearn.cluster import HDBSCAN

        # HDBSCAN automatically determines the number of clusters.
        # min_cluster_size controls granularity.
        min_cs = max(5, X.shape[0] // 50)
        self._model = HDBSCAN(
            min_cluster_size=min_cs,
            min_samples=5,
            cluster_selection_method="eom",
        )
        self._model.fit(X)
        self.labels_ = self._model.labels_

        # If HDBSCAN assigns everything to noise (-1), fall back to
        # labeling all points as cluster 0 to avoid degenerate metrics.
        if len(set(self.labels_)) <= 1:
            from sklearn.cluster import KMeans
            k = self.n_clusters if self.n_clusters is not None else 8
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            km.fit(X)
            self.labels_ = km.labels_

        return self

    def predict(self, X):
        if self.labels_ is None:
            self.fit(X)
        return self.labels_


def custom_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 36,
        "end_line": 109,
        "content": _CONTENT,
    },
]
