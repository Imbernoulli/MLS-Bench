"""R-Learner baseline for causal-treatment-effect.

R-Learner (Robinson decomposition): residualize both outcome and treatment,
then estimate CATE by minimizing a weighted loss on the residualized data.

Reference: Nie & Wager (2021). "Quasi-oracle estimation of heterogeneous
treatment effects." Biometrika.
"""

_FILE = "scikit-learn/custom_cate.py"

_CONTENT = """\
class CATEEstimator(BaseCATEEstimator):
    \"\"\"R-Learner: Robinson decomposition for CATE estimation.

    Based on the Robinson (1988) decomposition:
        Y - m(X) = (T - e(X)) * tau(X) + epsilon

    Steps:
    1. Cross-fit nuisance models:
       - m(X) = E[Y|X]  (marginal outcome model)
       - e(X) = P(T=1|X)  (propensity score)
    2. Compute residuals: Y_tilde = Y - m(X), T_tilde = T - e(X)
    3. Estimate tau(X) by minimizing: sum_i (Y_tilde_i - T_tilde_i * tau(X_i))^2
       This is equivalent to weighted least squares with weight T_tilde^2.
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

    def fit(self, X, T, Y):
        n = len(Y)

        # Cross-fit nuisance models
        kf = KFold(n_splits=5, shuffle=True, random_state=self._seed)
        m_hat = np.zeros(n)
        e_hat = np.zeros(n)

        for train_idx, val_idx in kf.split(X):
            # Outcome model E[Y|X]
            my = self._make_model_y()
            my.fit(X[train_idx], Y[train_idx])
            m_hat[val_idx] = my.predict(X[val_idx])

            # Propensity model P(T=1|X)
            mt = self._make_model_t()
            mt.fit(X[train_idx], T[train_idx])
            e_hat[val_idx] = mt.predict_proba(X[val_idx])[:, 1]

        # Clip propensity scores
        e_hat = np.clip(e_hat, 0.05, 0.95)

        # Residuals
        Y_tilde = Y - m_hat
        T_tilde = T - e_hat

        # R-Learner: pseudo-outcome = Y_tilde / T_tilde
        # Weight = T_tilde^2 (higher weight where treatment variation is larger)
        weights = T_tilde ** 2
        # Avoid division by zero
        safe_T = np.where(np.abs(T_tilde) > 0.01, T_tilde, np.sign(T_tilde) * 0.01 + 1e-8)
        pseudo = Y_tilde / safe_T

        # Clip extreme pseudo-outcomes
        q = np.percentile(np.abs(pseudo), 95)
        pseudo = np.clip(pseudo, -q, q)

        # Weighted regression for CATE
        # Use sample_weight = T_tilde^2 to prioritize informative samples
        self._cate_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.8, random_state=self._seed + 2,
        )
        self._cate_model.fit(X, pseudo, sample_weight=weights)
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
