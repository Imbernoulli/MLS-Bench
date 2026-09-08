"""RCD baseline — replaces run_causal_discovery with Repetitive Causal Discovery.

Reference: Maeda & Shimizu, "RCD: Repetitive causal discovery of linear
non-Gaussian acyclic models with latent confounders", AISTATS 2020.
"""

_FILE = "causal-learn/bench/custom_algorithm.py"

_RCD_FN = """\
def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    \"\"\"
    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables)
            B[i, j] != 0  means j -> i  (follows causal-learn convention)
    \"\"\"
    import numpy as np
    import itertools
    from scipy.stats import pearsonr, shapiro
    from sklearn.linear_model import LinearRegression
    from sklearn.utils import check_array
    from causallearn.search.FCMBased.lingam.hsic import hsic_test_gamma

    X = check_array(X)
    n_samples, n_features = X.shape

    # --- hyper-parameters (RCD defaults) ---
    max_explanatory_num = 2
    cor_alpha = 0.01
    ind_alpha = 0.01
    shapiro_alpha = 0.01
    bw_method = 'mdbs'

    # --- helpers ---
    def _resid_and_coef(endog_idx, exog_idcs):
        lr = LinearRegression()
        lr.fit(X[:, exog_idcs], X[:, endog_idx])
        resid = X[:, endog_idx] - lr.predict(X[:, exog_idcs])
        return resid, lr.coef_

    def _residual_matrix(U, H_U):
        if len(H_U) == 0:
            return X.copy()
        Y = np.zeros_like(X)
        for xj in U:
            Y[:, xj], _ = _resid_and_coef(xj, list(H_U))
        return Y

    def _is_non_gaussian(Y, U):
        for xj in U:
            if shapiro(Y[:, xj])[1] > shapiro_alpha:
                return False
        return True

    def _is_correlated(a, b):
        return pearsonr(a, b)[1] < cor_alpha

    def _is_independent(a, b):
        _, p = hsic_test_gamma(
            a.reshape(-1, 1), b.reshape(-1, 1), bw_method=bw_method)
        return p > ind_alpha

    def _exists_ancestor_in_U(M, U, xi, xj_list):
        for xj in xj_list:
            if xi in M[xj]:
                return True
        if set(xj_list) == set(xj_list) & M[xi]:
            return True
        return False

    def _is_independent_of_resid(Y, xi, xj_list):
        lr = LinearRegression()
        lr.fit(Y[:, xj_list], Y[:, xi])
        resid = Y[:, xi] - lr.predict(Y[:, xj_list])
        for xj in xj_list:
            if not _is_independent(resid, Y[:, xj]):
                return False
        return True

    # --- Step 1: extract ancestors ---
    M = [set() for _ in range(n_features)]
    l = 1
    hu_history = {}
    while True:
        changed = False
        for U in itertools.combinations(range(n_features), l + 1):
            U = sorted(U)

            # Common ancestors = intersection of all ancestor sets
            H_U = M[U[0]]
            for idx in U[1:]:
                H_U = H_U & M[idx]

            key = tuple(U)
            if key in hu_history and H_U == hu_history[key]:
                continue

            Y = _residual_matrix(U, H_U)

            if not _is_non_gaussian(Y, U):
                continue

            is_cor = True
            for xi, xj in itertools.combinations(U, 2):
                if not _is_correlated(Y[:, xi], Y[:, xj]):
                    is_cor = False
                    break
            if not is_cor:
                continue

            sink_set = []
            for xi in U:
                xj_list = [v for v in U if v != xi]
                if _exists_ancestor_in_U(M, U, xi, xj_list):
                    continue
                if _is_independent_of_resid(Y, xi, xj_list):
                    sink_set.append(xi)

            if len(sink_set) == 1:
                xi = sink_set[0]
                xj_list = [v for v in U if v != xi]
                new_ancestors = M[xi] | set(xj_list)
                if M[xi] != new_ancestors:
                    M[xi] = new_ancestors
                    changed = True

            hu_history[key] = H_U

        if changed:
            l = 1
        elif l < max_explanatory_num:
            l += 1
        else:
            break

    # --- Step 2: extract parents from ancestors ---
    P = [set() for _ in range(n_features)]
    for xi in range(n_features):
        for xj in M[xi]:
            others = M[xi] - {xj}
            if len(others) > 0:
                zi, _ = _resid_and_coef(xi, list(others))
            else:
                zi = X[:, xi]
            common = M[xi] & M[xj]
            if len(common) > 0:
                wj, _ = _resid_and_coef(xj, list(common))
            else:
                wj = X[:, xj]
            if _is_correlated(wj, zi):
                P[xi].add(xj)

    # --- Step 3: find confounder pairs ---
    C = [set() for _ in range(n_features)]
    for i, j in itertools.combinations(range(n_features), 2):
        if i in P[j] or j in P[i]:
            continue
        ri = X[:, i]
        if len(P[i]) > 0:
            lr = LinearRegression().fit(X[:, list(P[i])], X[:, i])
            ri = X[:, i] - lr.predict(X[:, list(P[i])])
        rj = X[:, j]
        if len(P[j]) > 0:
            lr = LinearRegression().fit(X[:, list(P[j])], X[:, j])
            rj = X[:, j] - lr.predict(X[:, list(P[j])])
        if _is_correlated(ri, rj):
            C[i].add(j)
            C[j].add(i)

    # --- Step 4: estimate adjacency matrix ---
    B = np.zeros((n_features, n_features), dtype='float64')
    for xi in range(n_features):
        xj_list = sorted(P[xi])
        if len(xj_list) == 0:
            continue
        _, coef = _resid_and_coef(xi, xj_list)
        for k, xj in enumerate(xj_list):
            B[xi, xj] = coef[k]

    # RCD marks confounder-affected pairs as NaN; we skip that for evaluation
    return B
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 6,
        "end_line": 13,
        "content": _RCD_FN,
    },
]
