"""Agglomerative Clustering baseline for ml-clustering-algorithm.

Hierarchical clustering with Ward linkage.
Reference: sklearn.cluster.AgglomerativeClustering
"""

_FILE = "scikit-learn/custom_clustering.py"

_CONTENT = """\

class CustomClustering(BaseEstimator, ClusterMixin):
    \"\"\"Agglomerative (hierarchical) clustering with Ward linkage.\"\"\"

    def __init__(self, n_clusters=None, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None

    def fit(self, X):
        from sklearn.cluster import AgglomerativeClustering

        k = self.n_clusters if self.n_clusters is not None else 8
        self._model = AgglomerativeClustering(
            n_clusters=k, linkage="ward"
        )
        self._model.fit(X)
        self.labels_ = self._model.labels_
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
