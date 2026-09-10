"""K-Means baseline for ml-clustering-algorithm.

Classic centroid-based clustering (Lloyd's algorithm).
Reference: sklearn.cluster.KMeans
"""

_FILE = "scikit-learn/custom_clustering.py"

_CONTENT = """\

class CustomClustering(BaseEstimator, ClusterMixin):
    \"\"\"K-Means clustering (Lloyd's algorithm).\"\"\"

    def __init__(self, n_clusters=None, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None
        self._model = None

    def fit(self, X):
        from sklearn.cluster import KMeans

        k = self.n_clusters if self.n_clusters is not None else 8
        self._model = KMeans(
            n_clusters=k, random_state=self.random_state, n_init=10, max_iter=300
        )
        self._model.fit(X)
        self.labels_ = self._model.labels_
        return self

    def predict(self, X):
        if self._model is None:
            self.fit(X)
        return self._model.predict(X)


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
