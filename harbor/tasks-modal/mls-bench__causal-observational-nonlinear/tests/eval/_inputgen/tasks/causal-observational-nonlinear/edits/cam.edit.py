"""CAM baseline -- replaces run_causal_discovery with Causal Additive Models.

Reference: Buehlmann et al., "CAM: Causal Additive Models, high-dimensional
order search and penalized regression", Annals of Statistics, 2014.

Implementation: CAM-inspired nonlinear heuristic using gradient-boosted
regressors for ordering and residual-correlation pruning.
"""

_FILE = "causal-learn/bench/custom_algorithm.py"

_CAM_FN = """\
def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    \"\"\"
    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables)
            B[i, j] != 0  means j -> i  (follows causal-learn convention)
    \"\"\"
    import os
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score

    n_samples, n_vars = X.shape
    seed = int(os.environ.get("SEED", "42"))

    # --- Step 1: Estimate causal ordering via greedy score-based search ---
    # For each variable, compute residual variance after nonlinear regression
    # on candidate parents. Variables with lower residual variance given
    # earlier variables are placed later in the ordering.

    remaining = list(range(n_vars))
    ordering = []

    while remaining:
        if len(ordering) == 0:
            # First variable: pick the one with highest marginal variance
            # (root causes tend to have variance = noise variance only)
            scores = []
            for j in remaining:
                scores.append(np.var(X[:, j]))
            # Pick the one with smallest variance (likely a root)
            best_idx = np.argmin(scores)
            ordering.append(remaining.pop(best_idx))
        else:
            # For each remaining var, fit nonlinear regression on current ordering
            best_score = np.inf
            best_var = None
            best_var_idx = None
            parents_X = X[:, ordering]
            for idx, j in enumerate(remaining):
                y = X[:, j]
                # Use gradient boosting as nonlinear regressor
                gbr = GradientBoostingRegressor(
                    n_estimators=50, max_depth=3, learning_rate=0.1,
                    random_state=seed, subsample=0.8
                )
                gbr.fit(parents_X, y)
                residuals = y - gbr.predict(parents_X)
                resid_var = np.var(residuals)
                if resid_var < best_score:
                    best_score = resid_var
                    best_var = j
                    best_var_idx = idx
            ordering.append(remaining.pop(best_var_idx))

    # --- Step 2: Preliminary adjacency via nonlinear regression along ordering ---
    B = np.zeros((n_vars, n_vars))
    for pos in range(1, len(ordering)):
        j = ordering[pos]
        candidate_parents = ordering[:pos]
        y = X[:, j]
        pa_X = X[:, candidate_parents]

        gbr = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=seed, subsample=0.8
        )
        gbr.fit(pa_X, y)
        importances = gbr.feature_importances_

        # Keep edges with importance above threshold
        threshold = 0.05
        for k, p in enumerate(candidate_parents):
            if importances[k] > threshold:
                B[j, p] = 1.0  # B[child, parent] = 1 means parent -> child

    # --- Step 3: Prune spurious edges via partial residual independence test ---
    for j in range(n_vars):
        parents = list(np.where(B[j, :] != 0)[0])
        if len(parents) <= 1:
            continue
        to_remove = []
        for p in parents:
            other_parents = [pp for pp in parents if pp != p]
            if len(other_parents) == 0:
                continue
            # Regress j on other parents
            gbr_j = GradientBoostingRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                random_state=seed
            )
            gbr_j.fit(X[:, other_parents], X[:, j])
            resid_j = X[:, j] - gbr_j.predict(X[:, other_parents])
            # Regress p on other parents
            gbr_p = GradientBoostingRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                random_state=seed
            )
            gbr_p.fit(X[:, other_parents], X[:, p])
            resid_p = X[:, p] - gbr_p.predict(X[:, other_parents])
            # Check correlation of partial residuals
            corr = np.abs(np.corrcoef(resid_j, resid_p)[0, 1])
            if corr < 0.05:
                to_remove.append(p)
        for p in to_remove:
            B[j, p] = 0.0

    return B
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 6,
        "end_line": 13,
        "content": _CAM_FN,
    },
]
