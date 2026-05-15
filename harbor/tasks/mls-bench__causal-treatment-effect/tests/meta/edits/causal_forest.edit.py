"""Causal Forest baseline for causal-treatment-effect.

Generalized Random Forest for heterogeneous treatment effect estimation.
Uses local centering (residualization) and honest splitting.

Reference: Athey, Tibshirani & Wager (2019). "Generalized Random Forests."
Annals of Statistics. (Original: Wager & Athey, 2018, JASA.)

Implementation uses econml.dml.CausalForestDML which combines:
- Orthogonal/doubly-robust moment conditions (Chernozhukov et al., 2018)
- Random forest-based CATE estimation with honest splitting
"""

_FILE = "scikit-learn/custom_cate.py"

_CONTENT = """\
class CATEEstimator(BaseCATEEstimator):
    \"\"\"Causal Forest (via econml CausalForestDML).

    Combines double machine learning (DML) for debiasing with
    generalized random forests for heterogeneous effect estimation.

    Steps:
    1. Cross-fit nuisance models: E[Y|X] and E[T|X]
    2. Compute residuals: Y_res = Y - E[Y|X], T_res = T - E[T|X]
    3. Fit a causal forest on residualized outcomes

    Falls back to a pure-sklearn implementation if econml is unavailable.
    \"\"\"

    def __init__(self):
        self._seed = int(os.environ.get("SEED", "42"))
        self._use_econml = True
        try:
            from econml.dml import CausalForestDML
            self._cf = CausalForestDML(
                model_y=GradientBoostingRegressor(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    min_samples_leaf=20, random_state=self._seed,
                ),
                model_t=GradientBoostingRegressor(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    min_samples_leaf=20, random_state=self._seed + 1,
                ),
                n_estimators=500,
                min_samples_leaf=5,
                max_depth=None,
                honest=True,
                inference=False,
                random_state=self._seed + 2,
                cv=3,
            )
        except ImportError:
            self._use_econml = False
            # Fallback: manual residualization + random forest
            self._model_y = GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                min_samples_leaf=20, random_state=self._seed,
            )
            self._model_t = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                min_samples_leaf=20, random_state=self._seed + 1,
            )
            self._cate_model = RandomForestRegressor(
                n_estimators=500, min_samples_leaf=5,
                max_features="sqrt", random_state=self._seed + 2,
            )

    def fit(self, X, T, Y):
        if self._use_econml:
            self._cf.fit(Y, T, X=X)
        else:
            # Manual DML: cross-fit residuals
            kf = KFold(n_splits=3, shuffle=True, random_state=self._seed)
            Y_res = np.zeros_like(Y)
            T_res = np.zeros_like(T, dtype=float)

            for train_idx, val_idx in kf.split(X):
                my = clone(self._model_y).fit(X[train_idx], Y[train_idx])
                mt = clone(self._model_t).fit(X[train_idx], T[train_idx])
                Y_res[val_idx] = Y[val_idx] - my.predict(X[val_idx])
                T_res[val_idx] = T[val_idx] - mt.predict_proba(X[val_idx])[:, 1]

            # R-Learner-style pseudo-outcome with stable divisor + sample
            # weighting so small |T_res| doesn't explode the fit.
            safe_T = np.where(np.abs(T_res) > 0.01, T_res, np.sign(T_res) * 0.01 + 1e-8)
            pseudo = Y_res / safe_T
            weights = T_res ** 2
            q = np.percentile(np.abs(pseudo), 95)
            pseudo = np.clip(pseudo, -q, q)
            self._cate_model.fit(X, pseudo, sample_weight=weights)
        return self

    def predict(self, X):
        if self._use_econml:
            return self._cf.effect(X).flatten()
        else:
            return self._cate_model.predict(X)
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
