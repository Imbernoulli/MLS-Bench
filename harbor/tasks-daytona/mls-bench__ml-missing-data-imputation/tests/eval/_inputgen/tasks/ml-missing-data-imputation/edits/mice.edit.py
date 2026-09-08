"""MICE (Multiple Imputation by Chained Equations) baseline — rigorous codebase edit ops.

Iterative imputation using chained equations with BayesianRidge as the estimator.
Each feature with missing values is modeled as a function of the other features
in a round-robin fashion, iterated until convergence.

Reference: sklearn.impute.IterativeImputer (van Buuren & Groothuis-Oudshoorn, 2011)
"""

_FILE = "scikit-learn/custom_imputation.py"

_MICE = """\
# ================================================================
# EDITABLE -- agent modifies this section (lines 36 to 142)
# ================================================================


class CustomImputer(BaseEstimator, TransformerMixin):
    \"\"\"MICE: Multiple Imputation by Chained Equations.

    Uses sklearn.impute.IterativeImputer with BayesianRidge estimator.
    Reference: van Buuren & Groothuis-Oudshoorn (2011).
    \"\"\"

    def __init__(self, random_state=42, max_iter=30):
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X, y=None):
        from sklearn.experimental import enable_iterative_imputer  # noqa
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge

        self._imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=self.max_iter,
            random_state=self.random_state,
            imputation_order="ascending",
            initial_strategy="mean",
            tol=1e-3,
        )
        self._imputer.fit(X)
        return self

    def transform(self, X):
        return self._imputer.transform(X)

    def fit_transform(self, X, y=None):
        from sklearn.experimental import enable_iterative_imputer  # noqa
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge

        self._imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=self.max_iter,
            random_state=self.random_state,
            imputation_order="ascending",
            initial_strategy="mean",
            tol=1e-3,
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
        "content": _MICE,
    },
]
