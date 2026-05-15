"""IPW (Inverse Propensity Weighting) baseline for causal-treatment-effect.

Horvitz-Thompson estimator using propensity scores for reweighting.
Estimates CATE by fitting a weighted outcome model after IPW adjustment.

Reference: Horvitz & Thompson (1952). "A generalization of sampling without
replacement from a finite universe." JASA.
Hirano, Imbens & Ridder (2003). "Efficient estimation of average treatment
effects using the estimated propensity score." Econometrica.
"""

_FILE = "scikit-learn/custom_cate.py"

_CONTENT = """\
class CATEEstimator(BaseCATEEstimator):
    \"\"\"IPW-based CATE estimator with propensity score weighting.

    1. Estimate propensity score e(X) = P(T=1|X) via logistic regression.
    2. Construct IPW pseudo-outcomes: Y_ipw = T*Y/e(X) - (1-T)*Y/(1-e(X)).
    3. Fit a regression model on X -> Y_ipw for CATE estimation.

    Clips propensity scores to [0.05, 0.95] for stability.
    \"\"\"

    def __init__(self):
        self._seed = int(os.environ.get("SEED", "42"))
        self._prop_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            min_samples_leaf=20, subsample=0.8, random_state=self._seed,
        )
        self._outcome_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=self._seed + 1,
        )

    def fit(self, X, T, Y):
        # Estimate propensity scores
        self._prop_model.fit(X, T)
        e_hat = self._prop_model.predict_proba(X)[:, 1]
        e_hat = np.clip(e_hat, 0.05, 0.95)

        # IPW pseudo-outcomes
        pseudo_outcome = T * Y / e_hat - (1 - T) * Y / (1 - e_hat)

        # Fit outcome model on pseudo-outcomes
        self._outcome_model.fit(X, pseudo_outcome)
        return self

    def predict(self, X):
        return self._outcome_model.predict(X)
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
