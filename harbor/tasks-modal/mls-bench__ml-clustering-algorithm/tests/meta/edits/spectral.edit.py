"""Spectral Clustering baseline for ml-clustering-algorithm.

Spectral clustering with RBF affinity kernel and automatic gamma selection.
SOTA graph-based method that can discover non-convex clusters.
Reference: Ng, Jordan, Weiss (2001) — "On Spectral Clustering: Analysis and
an Algorithm"; sklearn.cluster.SpectralClustering
"""

_FILE = "scikit-learn/custom_clustering.py"

_CONTENT = """\

class CustomClustering(BaseEstimator, ClusterMixin):
    \"\"\"Spectral Clustering with learned RBF affinity (Ng et al., 2001).\"\"\"

    def __init__(self, n_clusters=None, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None

    def fit(self, X):
        from sklearn.cluster import SpectralClustering

        k = self.n_clusters if self.n_clusters is not None else 8
        # Use nearest_neighbors affinity (robust across dimensions)
        self._model = SpectralClustering(
            n_clusters=k,
            affinity="nearest_neighbors",
            n_neighbors=10,
            random_state=self.random_state,
            n_init=10,
            assign_labels="kmeans",
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
