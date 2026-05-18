"""DR-Learner (Doubly Robust Learner) baseline for causal-treatment-effect.

Combines outcome modeling and propensity score weighting for doubly-robust
CATE estimation. Consistent if either the outcome model or propensity model
is correctly specified.

Reference: Kennedy (2023). "Towards optimal doubly robust estimation of
heterogeneous causal effects." Electronic Journal of Statistics.
Also: Chernozhukov et al. (2018). "Double/debiased machine learning for
treatment and structural parameters." Econometrics Journal.
"""

_FILE = "scikit-learn/custom_cate.py"

_CONTENT = """\
class CATEEstimator(BaseCATEEstimator):
    \"\"\"DR-Learner: Doubly Robust CATE estimation.

    Steps:
    1. Cross-fit nuisance models:
       - mu0(X) = E[Y|X, T=0], mu1(X) = E[Y|X, T=1]  (outcome models)
       - e(X) = P(T=1|X)  (propensity score)
    2. Compute doubly-robust pseudo-outcomes:
       phi(X) = mu1(X) - mu0(X)
              + T*(Y - mu1(X))/e(X)
              - (1-T)*(Y - mu0(X))/(1-e(X))
    3. Fit a final CATE model on X -> phi(X)
    \"\"\"

    def __init__(self):
        self._seed = int(os.environ.get("SEED", "42"))

    def _make_model_y(self):
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=20, subsample=0.8, random_state=self._seed,
        )

    def _make_model_t(self):
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            min_samples_leaf=20, subsample=0.8, random_state=self._seed + 1,
        )

    def _make_cate_model(self):
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.8, random_state=self._seed + 2,
        )

    def fit(self, X, T, Y):
        n = len(Y)

        # Cross-fit nuisance models
        kf = KFold(n_splits=5, shuffle=True, random_state=self._seed)
        mu0_hat = np.zeros(n)
        mu1_hat = np.zeros(n)
        e_hat = np.zeros(n)

        for train_idx, val_idx in kf.split(X):
            # Outcome models (separate for T=0 and T=1)
            mask0_train = T[train_idx] == 0
            mask1_train = T[train_idx] == 1

            m0 = self._make_model_y()
            m1 = self._make_model_y()

            if mask0_train.sum() > 5:
                m0.fit(X[train_idx[mask0_train]], Y[train_idx[mask0_train]])
                mu0_hat[val_idx] = m0.predict(X[val_idx])
            else:
                mu0_hat[val_idx] = Y[T == 0].mean() if (T == 0).sum() > 0 else Y.mean()

            if mask1_train.sum() > 5:
                m1.fit(X[train_idx[mask1_train]], Y[train_idx[mask1_train]])
                mu1_hat[val_idx] = m1.predict(X[val_idx])
            else:
                mu1_hat[val_idx] = Y[T == 1].mean() if (T == 1).sum() > 0 else Y.mean()

            # Propensity model
            mt = self._make_model_t()
            mt.fit(X[train_idx], T[train_idx])
            e_hat[val_idx] = mt.predict_proba(X[val_idx])[:, 1]

        # Clip propensity scores
        e_hat = np.clip(e_hat, 0.05, 0.95)

        # Doubly-robust pseudo-outcomes
        pseudo = (
            mu1_hat - mu0_hat
            + T * (Y - mu1_hat) / e_hat
            - (1 - T) * (Y - mu0_hat) / (1 - e_hat)
        )

        # Clip extreme pseudo-outcomes
        q = np.percentile(np.abs(pseudo), 97)
        pseudo = np.clip(pseudo, -q, q)

        # Fit CATE model on pseudo-outcomes
        self._cate_model = self._make_cate_model()
        self._cate_model.fit(X, pseudo)
        return self

    def predict(self, X):
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
