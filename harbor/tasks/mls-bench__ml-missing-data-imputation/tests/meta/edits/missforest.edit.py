"""MissForest baseline — rigorous codebase edit ops.

Iterative imputation using Random Forests. Each feature is imputed by training
a Random Forest on the observed values of that feature, using all other features
(with their current imputations) as predictors. Iterated until convergence.

Reference: Stekhoven & Buehlmann (2012), "MissForest — non-parametric missing
value imputation for mixed-type data", Bioinformatics 28(1):112-118.

This is a SOTA baseline that consistently outperforms simpler methods across
diverse datasets in benchmarks (2022-2025 comparative studies).
"""

_FILE = "scikit-learn/custom_imputation.py"

_MISSFOREST = """\
# ================================================================
# EDITABLE -- agent modifies this section (lines 36 to 142)
# ================================================================


class CustomImputer(BaseEstimator, TransformerMixin):
    \"\"\"MissForest: Iterative Random Forest imputation.

    Implements the MissForest algorithm (Stekhoven & Buehlmann, 2012):
    1. Initial imputation with column means
    2. For each iteration:
       a. Sort features by missingness (ascending)
       b. For each feature with missing values:
          - Train RandomForest on observed entries using all other features
          - Predict missing entries
       c. Check convergence (normalized difference < tol)
    3. Return when converged or max_iter reached

    Reference: Bioinformatics 28(1):112-118, 2012.
    \"\"\"

    def __init__(self, random_state=42, max_iter=10):
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_estimators = 100
        self.tol = 1e-4

    def fit(self, X, y=None):
        # Store the fitted state by running fit_transform internally
        self._X_fitted = X.copy()
        self._fit_transform_internal(X)
        return self

    def transform(self, X):
        return self._fit_transform_internal(X)

    def fit_transform(self, X, y=None):
        return self._fit_transform_internal(X)

    def _fit_transform_internal(self, X):
        from sklearn.ensemble import RandomForestRegressor

        X_imp = X.copy()
        n_samples, n_features = X_imp.shape

        # Step 1: Initial imputation with column means
        col_means = np.nanmean(X_imp, axis=0)
        for j in range(n_features):
            mask_j = np.isnan(X_imp[:, j])
            X_imp[mask_j, j] = col_means[j]

        # Identify which features have missing values and sort by missingness
        miss_count = np.isnan(X).sum(axis=0)
        features_with_missing = np.where(miss_count > 0)[0]
        # Sort by number of missing values (ascending)
        features_with_missing = features_with_missing[
            np.argsort(miss_count[features_with_missing])
        ]

        if len(features_with_missing) == 0:
            return X_imp

        # Step 2: Iterative imputation
        for iteration in range(self.max_iter):
            X_prev = X_imp.copy()

            for j in features_with_missing:
                # Observed and missing indices for feature j
                obs_mask = ~np.isnan(X[:, j])
                mis_mask = np.isnan(X[:, j])

                if mis_mask.sum() == 0:
                    continue

                # Predictor features (all except j)
                other_features = [k for k in range(n_features) if k != j]
                X_train = X_imp[obs_mask][:, other_features]
                y_train = X[obs_mask, j]  # Use original observed values
                X_pred = X_imp[mis_mask][:, other_features]

                # Train random forest and predict
                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_features="sqrt",
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                rf.fit(X_train, y_train)
                X_imp[mis_mask, j] = rf.predict(X_pred)

            # Step 3: Check convergence
            diff = np.sum((X_imp - X_prev) ** 2)
            denom = np.sum(X_imp ** 2)
            if denom > 0 and diff / denom < self.tol:
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
        "content": _MISSFOREST,
    },
]
