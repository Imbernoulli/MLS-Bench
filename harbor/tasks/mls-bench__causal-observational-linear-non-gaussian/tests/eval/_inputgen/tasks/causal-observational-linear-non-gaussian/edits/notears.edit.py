"""NOTEARS-linear baseline — replaces run_causal_discovery with NOTEARS.

Reference: Zheng, Aragam, Ravikumar, Xing, "DAGs with NO TEARS: Continuous
Optimization for Structure Learning", NeurIPS 2018.

Appropriate for linear SCMs without latent confounders. Uses continuous
optimization with a smooth acyclicity constraint h(W) = tr(exp(W*W)) - d.

Task-local expected ordering in the linear non-Gaussian leaderboard:
  DirectLiNGAM >= ICA-LiNGAM > NOTEARS-linear

NOTEARS does not explicitly exploit non-Gaussianity, so LiNGAM-family methods
are expected to be stronger on this task; NOTEARS remains a useful baseline
from a distinct method family (continuous optimization), providing
methodological diversity.
"""

_FILE = "causal-learn/bench/custom_algorithm.py"

_NOTEARS_FN = """\
def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    \"\"\"
    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables)
            B[i, j] != 0  means j -> i  (follows causal-learn convention)
    \"\"\"
    import numpy as np
    import scipy.linalg as sla
    from scipy.optimize import minimize
    from sklearn.utils import check_array

    X = check_array(X)
    n, d = X.shape

    # Reference defaults from Zheng et al. 2018 reference impl.
    lambda1 = 0.1
    max_iter = 100
    h_tol = 1e-8
    rho_max = 1e16
    w_threshold = 0.3

    def _loss_and_grad(W):
        # Squared-error regression loss: 1/(2n) * ||X - X W||^2
        R = X - X @ W
        loss = 0.5 / n * (R ** 2).sum()
        G = -1.0 / n * X.T @ R
        return loss, G

    def _h_and_grad(W):
        # h(W) = tr(exp(W*W)) - d  (Zheng 2018 smooth acyclicity)
        M = W * W
        E = sla.expm(M)
        h = np.trace(E) - d
        G = E.T * 2 * W
        return h, G

    def _obj(w_pm, rho, alpha):
        w_pm = w_pm.reshape(2, d * d)
        W = (w_pm[0] - w_pm[1]).reshape(d, d)
        loss, G_loss = _loss_and_grad(W)
        h, G_h = _h_and_grad(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w_pm.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g = np.concatenate([
            (G_smooth + lambda1).flatten(),
            (-G_smooth + lambda1).flatten(),
        ])
        return obj, g

    w_est = np.zeros(2 * d * d)
    rho, alpha, h = 1.0, 0.0, np.inf
    # Non-negative bounds; force diagonal to zero (no self-loops)
    bnds = [(0, 0) if (i == j) else (0, None)
            for _ in range(2) for i in range(d) for j in range(d)]

    for _ in range(max_iter):
        while rho < rho_max:
            sol = minimize(_obj, w_est, args=(rho, alpha),
                           method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            W_new = (w_new[:d * d] - w_new[d * d:]).reshape(d, d)
            h_new, _ = _h_and_grad(W_new)
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break

    W_final = (w_est[:d * d] - w_est[d * d:]).reshape(d, d)
    W_final[np.abs(W_final) < w_threshold] = 0.0

    # NOTEARS: W[i, j] != 0 means i -> j
    # causal-learn / this task: B[i, j] != 0 means j -> i. Transpose.
    return W_final.T
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 6,
        "end_line": 13,
        "content": _NOTEARS_FN,
    },
]
