"""Custom dimensionality reduction benchmark -- agent-editable template.

The agent modifies `CustomDimReduction` to implement a novel nonlinear
dimensionality reduction method.  The evaluation harness embeds three
datasets into 2D, then measures kNN accuracy, trustworthiness, and
continuity in the reduced space.
"""

import numpy as np
from numpy.typing import NDArray

# =====================================================================
# EDITABLE: implement CustomDimReduction below  (lines 15-59)
# =====================================================================
class CustomDimReduction:
    """Custom dimensionality reduction method.

    Must implement fit_transform(X) -> X_reduced.

    Parameters
    ----------
    n_components : int
        Target dimensionality (default 2).
    random_state : int or None
        Random seed for reproducibility.

    Notes
    -----
    You may use numpy and scipy (already installed).
    The method should work on datasets with 1000-70000 samples
    and 50-784 features, reducing to n_components=2 dimensions.
    It should preserve both local neighborhood structure and
    global data relationships.
    """

    def __init__(self, n_components: int = 2, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reduce dimensionality of X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            High-dimensional input data (standardized).

        Returns
        -------
        X_reduced : ndarray of shape (n_samples, n_components)
            Low-dimensional embedding.
        """
        # Default: random projection (poor baseline) -- replace with your design
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        projection = rng.randn(n_features, self.n_components)
        projection /= np.linalg.norm(projection, axis=0, keepdims=True)
        X_reduced = X @ projection
        return X_reduced
# =====================================================================
# END EDITABLE
# =====================================================================


# =====================================================================
# FIXED -- evaluation harness below (do not modify)
# =====================================================================
import argparse
import sys
import os
import time

from sklearn.datasets import fetch_openml, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score


def _trustworthiness(X_high, X_low, n_neighbors=7):
    """Compute trustworthiness: are nearest neighbors in the embedding
    also near in the original space?

    Measures preservation of local structure.
    Range: [0, 1], higher is better.
    """
    from scipy.spatial.distance import cdist

    n = X_high.shape[0]
    D_high = cdist(X_high, X_high, metric="sqeuclidean")
    D_low = cdist(X_low, X_low, metric="sqeuclidean")

    k = min(n_neighbors, n - 1)

    # Ranks in original space
    ranks_high = np.zeros_like(D_high, dtype=int)
    for i in range(n):
        order = np.argsort(D_high[i])
        ranks_high[i, order] = np.arange(n)

    # k-nearest neighbors in low-dimensional space
    trust_sum = 0.0
    for i in range(n):
        nn_low = np.argsort(D_low[i])[1 : k + 1]
        for j in nn_low:
            r = ranks_high[i, j]
            if r > k:
                trust_sum += r - k

    denom = n * k * (2 * n - 3 * k - 1)
    if denom == 0:
        return 1.0
    return 1.0 - (2.0 / denom) * trust_sum


def _continuity(X_high, X_low, n_neighbors=7):
    """Compute continuity: are nearest neighbors in the original space
    also near in the embedding?

    Measures whether the embedding tears apart nearby points.
    Range: [0, 1], higher is better.
    """
    from scipy.spatial.distance import cdist

    n = X_high.shape[0]
    D_high = cdist(X_high, X_high, metric="sqeuclidean")
    D_low = cdist(X_low, X_low, metric="sqeuclidean")

    k = min(n_neighbors, n - 1)

    # Ranks in low-dimensional space
    ranks_low = np.zeros_like(D_low, dtype=int)
    for i in range(n):
        order = np.argsort(D_low[i])
        ranks_low[i, order] = np.arange(n)

    # k-nearest neighbors in high-dimensional space
    cont_sum = 0.0
    for i in range(n):
        nn_high = np.argsort(D_high[i])[1 : k + 1]
        for j in nn_high:
            r = ranks_low[i, j]
            if r > k:
                cont_sum += r - k

    denom = n * k * (2 * n - 3 * k - 1)
    if denom == 0:
        return 1.0
    return 1.0 - (2.0 / denom) * cont_sum


def _knn_accuracy(X_reduced, y, n_neighbors=7, n_splits=3, seed=42):
    """Evaluate kNN classification accuracy in the reduced space.

    Uses stratified shuffle split to get mean accuracy.
    """
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=seed)
    accs = []
    for train_idx, test_idx in sss.split(X_reduced, y):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X_reduced[train_idx], y[train_idx])
        y_pred = clf.predict(X_reduced[test_idx])
        accs.append(accuracy_score(y[test_idx], y_pred))
    return float(np.mean(accs))


_MAX_SAMPLES = 5000  # subsample for tractability

_DATASETS = {
    "mnist": {
        "loader": lambda: fetch_openml(
            "mnist_784", version=1,
            data_home=os.environ.get("SKLEARN_DATA_HOME", None),
            parser="auto", as_frame=False,
        ),
        "preprocess": None,  # already numeric
    },
    "fashion_mnist": {
        "loader": lambda: fetch_openml(
            "Fashion-MNIST", version=1,
            data_home=os.environ.get("SKLEARN_DATA_HOME", None),
            parser="auto", as_frame=False,
        ),
        "preprocess": None,
    },
    "newsgroups": {
        "loader": lambda: None,  # special handling below
        "preprocess": "tfidf",
    },
}


def _load_dataset(name, seed=42):
    """Load and preprocess a dataset, returning (X, y) with at most _MAX_SAMPLES."""
    rng = np.random.RandomState(seed)

    if name == "newsgroups":
        cats = [
            "alt.atheism", "comp.graphics", "rec.sport.baseball",
            "sci.med", "talk.politics.guns",
        ]
        data = fetch_20newsgroups(
            subset="all", categories=cats,
            data_home=os.environ.get("SKLEARN_DATA_HOME", None),
            remove=("headers", "footers", "quotes"),
        )
        vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
        X_tfidf = vectorizer.fit_transform(data.data)
        # Reduce to 50 dims with TruncatedSVD for speed
        svd = TruncatedSVD(n_components=50, random_state=seed)
        X = svd.fit_transform(X_tfidf)
        y = np.array(data.target)
    else:
        info = _DATASETS[name]
        bunch = info["loader"]()
        X = np.array(bunch.data, dtype=np.float64)
        y = np.array(bunch.target)
        # Ensure y is integer-encoded
        if y.dtype.kind in ("U", "S", "O"):
            from sklearn.preprocessing import LabelEncoder
            y = LabelEncoder().fit_transform(y)

    # Subsample
    n = X.shape[0]
    if n > _MAX_SAMPLES:
        idx = rng.choice(n, _MAX_SAMPLES, replace=False)
        X, y = X[idx], y[idx]

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def evaluate(dataset_name: str, seed: int = 42, n_neighbors: int = 7):
    """Run dimensionality reduction + evaluation on a single dataset."""
    print(f"Loading dataset: {dataset_name} ...", flush=True)
    X, y = _load_dataset(dataset_name, seed=seed)
    print(f"  Shape: {X.shape}, classes: {len(np.unique(y))}", flush=True)

    print("Running custom dimensionality reduction ...", flush=True)
    t0 = time.time()
    reducer = CustomDimReduction(n_components=2, random_state=seed)
    X_reduced = reducer.fit_transform(X)
    elapsed = time.time() - t0
    print(f"TRAIN_METRICS dataset={dataset_name} elapsed={elapsed:.2f}s", flush=True)

    # Validate output shape
    assert X_reduced.shape == (X.shape[0], 2), (
        f"Expected shape ({X.shape[0]}, 2), got {X_reduced.shape}"
    )
    assert np.all(np.isfinite(X_reduced)), "Output contains NaN or Inf"

    # Compute metrics
    print("Computing kNN accuracy ...", flush=True)
    knn_acc = _knn_accuracy(X_reduced, y, n_neighbors=n_neighbors, seed=seed)

    print("Computing trustworthiness ...", flush=True)
    trust = _trustworthiness(X, X_reduced, n_neighbors=n_neighbors)

    print("Computing continuity ...", flush=True)
    cont = _continuity(X, X_reduced, n_neighbors=n_neighbors)

    print(
        f"DIMRED_METRICS knn_acc={knn_acc:.6f} "
        f"trustworthiness={trust:.6f} continuity={cont:.6f} "
        f"time={elapsed:.2f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Dimensionality reduction benchmark"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(_DATASETS.keys()),
        help="Dataset to evaluate on",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_neighbors", type=int, default=7, help="k for kNN/trust/cont")
    args = parser.parse_args()

    evaluate(args.dataset, seed=args.seed, n_neighbors=args.n_neighbors)


if __name__ == "__main__":
    main()
