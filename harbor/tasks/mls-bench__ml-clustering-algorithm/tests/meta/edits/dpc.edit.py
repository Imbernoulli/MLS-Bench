"""Density Peak Clustering (DPC) baseline for ml-clustering-algorithm.

SOTA density-based method that identifies cluster centers as points with
high local density and large distance to other high-density points.
Reference: Rodriguez & Laio (2014) — "Clustering by fast search and
find of density peaks", Science 344(6191).

This is a from-scratch implementation since sklearn does not include DPC.
"""

_FILE = "scikit-learn/custom_clustering.py"

_CONTENT = """\

class CustomClustering(BaseEstimator, ClusterMixin):
    \"\"\"Density Peak Clustering (Rodriguez & Laio, 2014).\"\"\"

    def __init__(self, n_clusters=None, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None

    def fit(self, X):
        from scipy.spatial.distance import pdist, squareform

        n = X.shape[0]
        # Compute pairwise distance matrix
        dist_condensed = pdist(X)
        dist_matrix = squareform(dist_condensed)

        # Estimate dc (cutoff distance) so that avg neighbor count ~ 1-2% of data
        percent = 2.0
        sorted_dists = np.sort(dist_condensed)
        position = int(len(sorted_dists) * percent / 100.0)
        dc = sorted_dists[max(position, 1)]

        # Compute local density rho using Gaussian kernel
        rho = np.sum(np.exp(-(dist_matrix / dc) ** 2), axis=1) - 1.0

        # Compute delta (min distance to higher-density point) and nearest neighbor
        delta = np.full(n, np.inf)
        nearest_higher = np.full(n, -1, dtype=int)
        rho_order = np.argsort(-rho)

        for i in range(1, n):
            idx = rho_order[i]
            higher_density_points = rho_order[:i]
            dists_to_higher = dist_matrix[idx, higher_density_points]
            min_idx = np.argmin(dists_to_higher)
            delta[idx] = dists_to_higher[min_idx]
            nearest_higher[idx] = higher_density_points[min_idx]

        # Highest density point: delta = max distance
        delta[rho_order[0]] = np.max(dist_matrix[rho_order[0]])

        # Select cluster centers: top-k by gamma = rho * delta
        gamma = rho * delta
        k = self.n_clusters if self.n_clusters is not None else 8
        center_indices = np.argsort(-gamma)[:k]

        # Assign labels by propagating from centers in density order
        labels = np.full(n, -1, dtype=int)
        for cluster_id, center in enumerate(center_indices):
            labels[center] = cluster_id

        for idx in rho_order:
            if labels[idx] == -1 and nearest_higher[idx] != -1:
                labels[idx] = labels[nearest_higher[idx]]

        # Handle any remaining unassigned points (assign to nearest center)
        unassigned = np.where(labels == -1)[0]
        if len(unassigned) > 0:
            for idx in unassigned:
                dists_to_centers = dist_matrix[idx, center_indices]
                labels[idx] = np.argmin(dists_to_centers)

        self.labels_ = labels
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
