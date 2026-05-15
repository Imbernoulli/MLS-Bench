"""MICEForest baseline — rigorous codebase edit ops.

MICE with gradient-boosted trees (ExtraTreesRegressor as proxy for LightGBM).
Combines the iterative chained-equations framework of MICE with tree-based
estimators, approximating the miceforest library's approach.

Reference: AnotherSamWilson/miceforest (2022-2025), "Multiple Imputation with
LightGBM in Python". Uses predictive mean matching for improved quality.

This is a SOTA baseline combining MICE's iterative framework with powerful
tree-based models and predictive mean matching.
"""

_FILE = "scikit-learn/custom_imputation.py"

_MICEFOREST = """\
# ================================================================
# EDITABLE -- agent modifies this section (lines 36 to 142)
# ================================================================


class CustomImputer(BaseEstimator, TransformerMixin):
    \"\"\"MICEForest: MICE with ExtraTrees + Predictive Mean Matching.

    Approximates miceforest (AnotherSamWilson, 2022-2025):
    - Uses ExtraTreesRegressor as the chained-equations estimator
      (proxy for LightGBM, which miceforest uses internally)
    - Applies predictive mean matching (PMM): instead of using raw
      predictions, sample from observed values with closest predicted values
    - Iterative imputation with convergence check

    Hyperparameters:
    - n_estimators=100 (trees per feature model)
    - pmm_k=5 (number of nearest candidates for predictive mean matching)
    - max_iter=10 (maximum imputation rounds)
    \"\"\"

    def __init__(self, random_state=42, max_iter=10):
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_estimators = 100
        self.pmm_k = 5

    def fit(self, X, y=None):
        self._rng = np.random.RandomState(self.random_state)
        self._n_features = X.shape[1]
        # Store training data for PMM
        self._X_train = X.copy()
        return self

    def transform(self, X):
        return self._impute(X)

    def fit_transform(self, X, y=None):
        self._rng = np.random.RandomState(self.random_state)
        self._n_features = X.shape[1]
        self._X_train = X.copy()
        return self._impute(X)

    def _impute(self, X):
        from sklearn.ensemble import ExtraTreesRegressor

        X_imp = X.copy()
        n_samples, n_features = X_imp.shape

        # Initial imputation with column means
        col_means = np.nanmean(X_imp, axis=0)
        for j in range(n_features):
            mask_j = np.isnan(X_imp[:, j])
            X_imp[mask_j, j] = col_means[j]

        # Sort features by missingness (ascending)
        miss_count = np.isnan(X).sum(axis=0)
        features_with_missing = np.where(miss_count > 0)[0]
        features_with_missing = features_with_missing[
            np.argsort(miss_count[features_with_missing])
        ]

        if len(features_with_missing) == 0:
            return X_imp

        for iteration in range(self.max_iter):
            X_prev = X_imp.copy()

            for j in features_with_missing:
                obs_mask = ~np.isnan(X[:, j])
                mis_mask = np.isnan(X[:, j])

                if mis_mask.sum() == 0:
                    continue

                other_features = [k for k in range(n_features) if k != j]
                X_train = X_imp[obs_mask][:, other_features]
                y_train = X[obs_mask, j]
                X_pred = X_imp[mis_mask][:, other_features]

                # Fit ExtraTrees
                et = ExtraTreesRegressor(
                    n_estimators=self.n_estimators,
                    max_features="sqrt",
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                et.fit(X_train, y_train)

                # Predictive mean matching
                y_pred_obs = et.predict(X_train)  # Predictions on observed
                y_pred_mis = et.predict(X_pred)    # Predictions on missing

                # For each missing value, find pmm_k nearest observed predictions
                # and sample one of their actual values
                for i, pred_val in enumerate(y_pred_mis):
                    dists = np.abs(y_pred_obs - pred_val)
                    k = min(self.pmm_k, len(dists))
                    nearest_idx = np.argpartition(dists, k)[:k]
                    chosen = self._rng.choice(nearest_idx)
                    X_imp[np.where(mis_mask)[0][i], j] = y_train[chosen]

            # Check convergence
            diff = np.sum((X_imp - X_prev) ** 2)
            denom = np.sum(X_imp ** 2)
            if denom > 0 and diff / denom < 1e-4:
                break

        return X_imp


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
        "content": _MICEFOREST,
    },
]
