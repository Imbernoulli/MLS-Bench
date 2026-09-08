"""HC (Hill-Climbing) baseline — replaces run_causal_discovery with greedy HC.

Reference: Heckerman et al., "Learning Bayesian networks: The combination of
knowledge and statistical data", 1995.
Uses BDeu score with greedy add/delete/reverse edge operators.
"""

_FILE = "causal-bnlearn/bench/custom_algorithm.py"

_HC_FN = """\
def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    \"\"\"
    Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
    \"\"\"
    from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
    from causallearn.score.LocalScoreFunction import local_score_BDeu
    from causallearn.utils.DAG2CPDAG import dag2cpdag

    N = X.shape[1]
    # Pass parameters=None so local_score_BDeu auto-computes r_i_map from data
    score_func = LocalScoreClass(
        data=X, local_score_fun=local_score_BDeu, parameters=None
    )

    nodes = [GraphNode(f"X{i + 1}") for i in range(N)]
    adj = np.zeros((N, N), dtype=int)

    # Cache local scores (one per node)
    local_scores = np.zeros(N)
    for j in range(N):
        local_scores[j] = score_func.score(j, [])

    def _has_path(src, tgt):
        \"\"\"DFS check: is there a directed path from src to tgt in adj?\"\"\"
        visited = set()
        stack = [src]
        while stack:
            node = stack.pop()
            if node == tgt:
                return True
            if node in visited:
                continue
            visited.add(node)
            for c in np.where(adj[node] == 1)[0]:
                if int(c) not in visited:
                    stack.append(int(c))
        return False

    # Greedy hill-climbing: add / delete / reverse
    improved = True
    while improved:
        improved = False
        best_delta = 0.0
        best_op = None

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                if adj[i, j] == 0 and adj[j, i] == 0:
                    # --- Try ADD i -> j (only if no cycle) ---
                    if not _has_path(j, i):
                        pj_new = sorted(
                            np.where(adj[:, j] == 1)[0].tolist() + [i]
                        )
                        new_sj = score_func.score(j, pj_new)
                        delta = new_sj - local_scores[j]
                        if delta < best_delta - 1e-6:
                            best_delta = delta
                            best_op = ("add", i, j)

                elif adj[i, j] == 1:
                    # --- Try DELETE i -> j ---
                    pj_new = [
                        p for p in np.where(adj[:, j] == 1)[0] if p != i
                    ]
                    new_sj = score_func.score(j, sorted(pj_new))
                    delta = new_sj - local_scores[j]
                    if delta < best_delta - 1e-6:
                        best_delta = delta
                        best_op = ("delete", i, j)

                    # --- Try REVERSE i -> j  to  j -> i ---
                    adj[i, j] = 0  # temporarily remove
                    if not _has_path(i, j):
                        pj_del = sorted(
                            np.where(adj[:, j] == 1)[0].tolist()
                        )
                        new_sj = score_func.score(j, pj_del)
                        pi_new = sorted(
                            np.where(adj[:, i] == 1)[0].tolist() + [j]
                        )
                        new_si = score_func.score(i, pi_new)
                        delta = (
                            (new_sj - local_scores[j])
                            + (new_si - local_scores[i])
                        )
                        if delta < best_delta - 1e-6:
                            best_delta = delta
                            best_op = ("reverse", i, j)
                    adj[i, j] = 1  # restore

        if best_op is not None:
            op_type, i, j = best_op
            if op_type == "add":
                adj[i, j] = 1
            elif op_type == "delete":
                adj[i, j] = 0
            elif op_type == "reverse":
                adj[i, j] = 0
                adj[j, i] = 1
            # Recompute affected local scores
            local_scores[j] = score_func.score(
                j, sorted(np.where(adj[:, j] == 1)[0].tolist())
            )
            if op_type == "reverse":
                local_scores[i] = score_func.score(
                    i, sorted(np.where(adj[:, i] == 1)[0].tolist())
                )
            improved = True

    # Build GeneralGraph from learned DAG
    G = GeneralGraph(nodes)
    for i in range(N):
        for j in range(N):
            if adj[i, j] == 1:
                G.add_directed_edge(nodes[i], nodes[j])

    G = dag2cpdag(G)
    return G
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 8,
        "end_line": 14,
        "content": _HC_FN,
    },
]
