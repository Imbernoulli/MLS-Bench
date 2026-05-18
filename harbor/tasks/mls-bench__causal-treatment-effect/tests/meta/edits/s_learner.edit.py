"""S-Learner baseline for causal-treatment-effect.

S-Learner (Single model): fit one model on (X, T) -> Y, then
predict CATE as mu(X, T=1) - mu(X, T=0).

Reference: Kunzel et al. (2019). "Metalearners for estimating heterogeneous
treatment effects using machine learning." PNAS.
"""

_FILE = "scikit-learn/custom_cate.py"

_CONTENT = """\
class CATEEstimator(BaseCATEEstimator):
    \"\"\"S-Learner: single model approach to CATE estimation.

    Fits a single outcome model mu(X, T) on the combined data, then
    estimates CATE as mu(X, 1) - mu(X, 0).
    Uses GradientBoostingRegressor as the base learner for flexibility.
    \"\"\"

    def __init__(self):
        self._seed = int(os.environ.get("SEED", "42"))
        self._model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=self._seed,
        )

    def fit(self, X, T, Y):
        n, p = X.shape
        XT = np.column_stack([X, T.reshape(-1, 1)])
        self._model.fit(XT, Y)
        return self

    def predict(self, X):
        n = X.shape[0]
        X1 = np.column_stack([X, np.ones((n, 1))])
        X0 = np.column_stack([X, np.zeros((n, 1))])
        return self._model.predict(X1) - self._model.predict(X0)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 344,
        "end_line": 416,
        "content": _CONTENT,
    },
]
