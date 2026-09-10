"""NOTEARS nonlinear baseline for run_causal_discovery.

Reference: Zheng et al., "DAGs with NO TEARS: Continuous Optimization for
Structure Learning", NeurIPS 2018.
Zheng et al., "Learning Sparse Nonparametric DAGs", AISTATS 2020.

Implementation: NOTEARS continuous acyclicity optimization followed by
nonlinear regression refinement for the task harness.
"""

_FILE = "causal-learn/bench/custom_algorithm.py"

_NOTEARS_MLP_FN = """\
def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    \"\"\"
    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables)
            B[i, j] != 0  means j -> i  (follows causal-learn convention)
    \"\"\"
    import os
    import numpy as np
    from scipy.optimize import minimize

    n_samples, n_vars = X.shape
    seed = int(os.environ.get("SEED", "42"))

    # --- Hyperparameters ---
    max_iter = 30
    h_tol = 1e-8
    rho_max = 1e+16
    w_threshold = 0.3
    lambda1 = 0.01  # L1 penalty

    # Standardize data
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # --- NOTEARS (linear) on the data ---
    # Formulation: minimize 0.5/n * ||X - X W||^2_F + lambda1 * |W|_1
    # subject to h(W) = tr(e^{W o W}) - d = 0

    def _h(W):
        \"\"\"Acyclicity constraint: h(W) = tr(e^{W o W}) - d.\"\"\"
        M = W * W
        # Matrix exponential trace via power series (consistent with _h_grad)
        expm_M = np.eye(n_vars)
        power = np.eye(n_vars)
        for k in range(1, 12):
            power = power @ M / k
            expm_M += power
        return np.trace(expm_M) - n_vars

    def _h_grad(W):
        \"\"\"Gradient of h w.r.t. W.\"\"\"
        M = W * W
        # expm(M) via series (10 terms)
        expm_M = np.eye(n_vars)
        power = np.eye(n_vars)
        for k in range(1, 12):
            power = power @ M / k
            expm_M += power
        return 2 * W * expm_M

    def _loss_and_grad(W_flat, rho, alpha):
        W = W_flat.reshape(n_vars, n_vars)
        # Zero diagonal (no self-loops)
        np.fill_diagonal(W, 0)

        # MSE loss: 0.5/n * ||X - XW||^2
        R = X_std - X_std @ W  # (n, d)
        loss = 0.5 / n_samples * np.sum(R ** 2)
        # Gradient of MSE w.r.t. W
        G_mse = -1.0 / n_samples * (X_std.T @ R)  # (d, d)

        # L1 penalty
        l1_loss = lambda1 * np.sum(np.abs(W))
        G_l1 = lambda1 * np.sign(W)

        # Acyclicity
        h_val = _h(W)
        G_h = _h_grad(W)

        total_loss = loss + l1_loss + 0.5 * rho * h_val ** 2 + alpha * h_val
        G_total = G_mse + G_l1 + (rho * h_val + alpha) * G_h

        # Zero diagonal gradient
        np.fill_diagonal(G_total, 0)

        return total_loss, G_total.ravel()

    # --- Augmented Lagrangian ---
    # Small random init to break symmetry (zeros can stall the optimizer)
    rng = np.random.RandomState(seed)
    W_est = rng.randn(n_vars, n_vars) * 0.01
    np.fill_diagonal(W_est, 0)
    rho = 1.0
    alpha_dual = 0.0
    h_prev = np.inf

    for _ in range(max_iter):
        result = minimize(
            lambda w: _loss_and_grad(w, rho, alpha_dual),
            W_est.ravel(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500}
        )
        W_est = result.x.reshape(n_vars, n_vars)
        np.fill_diagonal(W_est, 0)

        h_new = _h(W_est)
        if h_new > 0.25 * h_prev:
            rho *= 10.0
        alpha_dual += rho * h_new
        h_prev = h_new

        if abs(h_new) < h_tol or rho > rho_max:
            break

    # --- Now do nonlinear refinement: for each variable, use kernel regression ---
    # Use the linear NOTEARS skeleton and refine with nonlinear regression
    from sklearn.ensemble import GradientBoostingRegressor

    # Threshold the linear result
    W_abs = np.abs(W_est)
    W_abs[W_abs < w_threshold] = 0.0

    # Refine: for each node, check if candidate parents improve nonlinear fit
    B = np.zeros((n_vars, n_vars))
    for j in range(n_vars):
        # Candidate parents from linear NOTEARS
        candidates = np.where(W_abs[:, j] > 0)[0].tolist()
        # Also add strong linear correlations as candidates
        for i in range(n_vars):
            if i == j:
                continue
            corr = np.abs(np.corrcoef(X_std[:, i], X_std[:, j])[0, 1])
            if corr > 0.15 and i not in candidates:
                candidates.append(i)
        if not candidates:
            continue

        # Nonlinear regression from candidates to j
        gbr = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=seed, subsample=0.8
        )
        gbr.fit(X_std[:, candidates], X_std[:, j])
        importances = gbr.feature_importances_

        for k, p in enumerate(candidates):
            if importances[k] > 0.05:
                B[j, p] = 1.0

    # Enforce DAG by removing cycles using topological ordering from linear NOTEARS
    # Use causal order from W_est
    order_score = np.sum(np.abs(W_est), axis=0)  # higher = more downstream
    topo_order = np.argsort(order_score)  # ascending = more root-like first
    rank = np.zeros(n_vars, dtype=int)
    for idx, node in enumerate(topo_order):
        rank[node] = idx

    # Remove edges that violate topological ordering
    for i in range(n_vars):
        for j in range(n_vars):
            if B[i, j] != 0 and rank[j] >= rank[i]:
                # j -> i but j is downstream of i: remove
                B[i, j] = 0.0

    return B
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 6,
        "end_line": 13,
        "content": _NOTEARS_MLP_FN,
    },
]
