"""DirectLiNGAM baseline — replaces run_causal_discovery with DirectLiNGAM.

Reference: Shimizu et al., "DirectLiNGAM: A direct method for learning a
linear non-Gaussian structural equation model", JMLR 2011.
"""

_FILE = "causal-learn/bench/custom_algorithm.py"

_DIRECTLINGAM_FN = """\
def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    \"\"\"
    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables)
            B[i, j] != 0  means j -> i  (follows causal-learn convention)
    \"\"\"
    import os
    import numpy as np
    from sklearn.utils import check_array
    from causallearn.search.FCMBased.lingam.base import _BaseLiNGAM

    X = check_array(X)
    seed = int(os.environ.get("SEED", "42"))
    n_features = X.shape[1]

    # Core DirectLiNGAM steps (from causallearn.search.FCMBased.lingam.direct_lingam)
    def _residual(xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        return xi - (np.cov(xi, xj)[0, 1] / np.var(xj)) * xj

    def _entropy(u: np.ndarray) -> float:
        k1 = 79.047
        k2 = 7.4129
        gamma = 0.37457
        return (1 + np.log(2 * np.pi)) / 2 - \
               k1 * (np.mean(np.log(np.cosh(u))) - gamma) ** 2 - \
               k2 * (np.mean(u * np.exp((-u ** 2) / 2))) ** 2

    def _diff_mutual_info(
        xi_std: np.ndarray,
        xj_std: np.ndarray,
        ri_j: np.ndarray,
        rj_i: np.ndarray,
    ) -> float:
        return (_entropy(xj_std) + _entropy(ri_j / np.std(ri_j))) - \
               (_entropy(xi_std) + _entropy(rj_i / np.std(rj_i)))

    def _search_causal_order(X_work: np.ndarray, U: np.ndarray) -> int:
        if len(U) == 1:
            return int(U[0])
        M_list = []
        for i in U:
            M = 0.0
            for j in U:
                if i == j:
                    continue
                xi_std = (X_work[:, i] - np.mean(X_work[:, i])) / np.std(X_work[:, i])
                xj_std = (X_work[:, j] - np.mean(X_work[:, j])) / np.std(X_work[:, j])
                ri_j = _residual(xi_std, xj_std)
                rj_i = _residual(xj_std, xi_std)
                M += np.min([0.0, _diff_mutual_info(xi_std, xj_std, ri_j, rj_i)]) ** 2
            M_list.append(-1.0 * M)
        return int(U[np.argmax(M_list)])

    U = np.arange(n_features)
    K = []
    X_work = np.copy(X)
    for _ in range(n_features):
        m = _search_causal_order(X_work, U)
        for i in U:
            if i != m:
                X_work[:, i] = _residual(X_work[:, i], X_work[:, m])
        K.append(m)
        U = U[U != m]

    class _LocalDirectLiNGAM(_BaseLiNGAM):
        def fit(self, X):
            return self

    model = _LocalDirectLiNGAM(random_state=seed)
    model._causal_order = K
    model._estimate_adjacency_matrix(X)
    return model.adjacency_matrix_
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 6,
        "end_line": 13,
        "content": _DIRECTLINGAM_FN,
    },
]
