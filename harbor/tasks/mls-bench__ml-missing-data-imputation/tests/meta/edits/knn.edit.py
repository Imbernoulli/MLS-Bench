"""KNN Imputation baseline — rigorous codebase edit ops.

Use K-Nearest Neighbors to impute missing values. For each missing entry,
find K nearest neighbors using observed features, then impute with the
weighted mean of neighbors' values.

Reference: sklearn.impute.KNNImputer (Troyanskaya et al., 2001)
"""

_FILE = "scikit-learn/custom_imputation.py"

_KNN = """\
# ================================================================
# EDITABLE -- agent modifies this section (lines 36 to 142)
# ================================================================


class CustomImputer(BaseEstimator, TransformerMixin):
    \"\"\"KNN Imputation: impute using K-nearest neighbors.

    Uses sklearn.impute.KNNImputer with n_neighbors=5, distance weighting.
    Reference: Troyanskaya et al. (2001).
    \"\"\"

    def __init__(self, random_state=42, max_iter=10):
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_neighbors = 5

    def fit(self, X, y=None):
        from sklearn.impute import KNNImputer
        self._imputer = KNNImputer(
            n_neighbors=self.n_neighbors,
            weights="distance",
        )
        self._imputer.fit(X)
        return self

    def transform(self, X):
        return self._imputer.transform(X)

    def fit_transform(self, X, y=None):
        from sklearn.impute import KNNImputer
        self._imputer = KNNImputer(
            n_neighbors=self.n_neighbors,
            weights="distance",
        )
        return self._imputer.fit_transform(X)


def compute_feature_correlations(X):
    n_features = X.shape[1]
    corr = np.eye(n_features)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            mask = ~(np.isnan(X[:, i]) | np.isnan(X[:, j]))
            if mask.sum() > 2:
                c = np.corrcoef(X[mask, i], X[mask, j])[0, 1]
                corr[i, j] = corr[j, i] = c if not np.isnan(c) else 0.0
    return corr

"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 36,
        "end_line": 131,
        "content": _KNN,
    },
]
