"""T-Learner baseline for causal-treatment-effect.

T-Learner (Two model): fit separate outcome models for treated and control,
then predict CATE as mu1(X) - mu0(X).

Reference: Kunzel et al. (2019). "Metalearners for estimating heterogeneous
treatment effects using machine learning." PNAS.
"""

_FILE = "scikit-learn/custom_cate.py"

_CONTENT = """\
class CATEEstimator(BaseCATEEstimator):
    \"\"\"T-Learner: two separate models for treated and control groups.

    Fits mu0(X) on control data and mu1(X) on treated data, then
    estimates CATE as mu1(X) - mu0(X).
    Uses GradientBoostingRegressor for both models.
    \"\"\"

    def __init__(self):
        self._seed = int(os.environ.get("SEED", "42"))
        self._model0 = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=self._seed,
        )
        self._model1 = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=self._seed + 1,
        )

    def fit(self, X, T, Y):
        mask0 = T == 0
        mask1 = T == 1
        self._model0.fit(X[mask0], Y[mask0])
        self._model1.fit(X[mask1], Y[mask1])
        return self

    def predict(self, X):
        return self._model1.predict(X) - self._model0.predict(X)
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
