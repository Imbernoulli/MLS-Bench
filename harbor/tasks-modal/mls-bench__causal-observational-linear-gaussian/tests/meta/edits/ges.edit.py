"""GES baseline — replaces run_causal_discovery with GES.

Reference: Chickering, "Optimal structure identification with greedy search", 2002.
"""

_FILE = "causal-learn/bench/custom_algorithm.py"

_GES_FN = """\
def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    \"\"\"
    Input:  X of shape (n_samples, n_variables)
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
    \"\"\"
    from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
    from causallearn.score.LocalScoreFunction import local_score_BIC_from_cov
    from causallearn.graph.Endpoint import Endpoint
    from causallearn.utils.GESUtils import (
        Combinatorial, score_g,
        insert_validity_test1, insert_validity_test2, find_subset_include,
        insert_changed_score, insert,
        delete_validity_test, delete_changed_score, delete,
    )
    from causallearn.utils.DAG2CPDAG import dag2cpdag
    from causallearn.utils.PDAG2DAG import pdag2dag

    N = X.shape[1]
    maxP = N / 2
    parameters = {"lambda_value": 2}
    score_func = LocalScoreClass(
        data=X, local_score_fun=local_score_BIC_from_cov, parameters=parameters
    )

    nodes = [GraphNode("X%d" % (i + 1)) for i in range(N)]
    G = GeneralGraph(nodes)
    score = score_g(X, G, score_func, parameters)
    G = pdag2dag(G)
    G = dag2cpdag(G)

    # Forward greedy search (Insert operators)
    record_local_score = [[] for _ in range(N)]
    score_new = score
    while True:
        score = score_new
        min_chscore = 1e7
        min_desc = []
        for i in range(N):
            for j in range(N):
                if (
                    G.graph[i, j] == Endpoint.NULL.value
                    and G.graph[j, i] == Endpoint.NULL.value
                    and i != j
                    and len(np.where(G.graph[j, :] == Endpoint.ARROW.value)[0]) <= maxP
                ):
                    Tj = np.intersect1d(
                        np.where(G.graph[:, j] == Endpoint.TAIL.value)[0],
                        np.where(G.graph[j, :] == Endpoint.TAIL.value)[0],
                    )
                    Ti = np.union1d(
                        np.where(G.graph[:, i] != Endpoint.NULL.value)[0],
                        np.where(G.graph[i, :] != Endpoint.NULL.value)[0],
                    )
                    NTi = np.setdiff1d(np.arange(N), Ti)
                    T0 = np.intersect1d(Tj, NTi)
                    sub = Combinatorial(T0.tolist())
                    S = np.zeros(len(sub))
                    for k in range(len(sub)):
                        if S[k] < 2:
                            V1 = insert_validity_test1(G, i, j, sub[k])
                            if V1:
                                if not S[k]:
                                    V2 = insert_validity_test2(G, i, j, sub[k])
                                else:
                                    V2 = 1
                                if V2:
                                    Idx = find_subset_include(sub[k], sub)
                                    S[np.where(Idx == 1)] = 1
                                    chscore, desc, record_local_score = insert_changed_score(
                                        X, G, i, j, sub[k],
                                        record_local_score, score_func, parameters,
                                    )
                                    if chscore < min_chscore:
                                        min_chscore = chscore
                                        min_desc = desc
                            else:
                                Idx = find_subset_include(sub[k], sub)
                                S[np.where(Idx == 1)] = 2
        if len(min_desc) != 0:
            score_new = score + min_chscore
            if score - score_new <= 0:
                break
            G = insert(G, min_desc[0], min_desc[1], min_desc[2])
            G = pdag2dag(G)
            G = dag2cpdag(G)
        else:
            score_new = score
            break

    # Backward greedy search (Delete operators)
    score_new = score
    while True:
        score = score_new
        min_chscore = 1e7
        min_desc = []
        for i in range(N):
            for j in range(N):
                if (
                    G.graph[j, i] == Endpoint.TAIL.value
                    and G.graph[i, j] == Endpoint.TAIL.value
                ) or G.graph[j, i] == Endpoint.ARROW.value:
                    Hj = np.intersect1d(
                        np.where(G.graph[:, j] == Endpoint.TAIL.value)[0],
                        np.where(G.graph[j, :] == Endpoint.TAIL.value)[0],
                    )
                    Hi = np.union1d(
                        np.where(G.graph[i, :] != Endpoint.NULL.value)[0],
                        np.where(G.graph[:, i] != Endpoint.NULL.value)[0],
                    )
                    H0 = np.intersect1d(Hj, Hi)
                    sub = Combinatorial(H0.tolist())
                    S = np.ones(len(sub))
                    for k in range(len(sub)):
                        if S[k] == 1:
                            V = delete_validity_test(G, i, j, sub[k])
                            if V:
                                Idx = find_subset_include(sub[k], sub)
                                S[np.where(Idx == 1)] = 2
                        else:
                            V = 1
                        if V:
                            chscore, desc, record_local_score = delete_changed_score(
                                X, G, i, j, sub[k],
                                record_local_score, score_func, parameters,
                            )
                            if chscore < min_chscore:
                                min_chscore = chscore
                                min_desc = desc
        if len(min_desc) != 0:
            score_new = score + min_chscore
            if score - score_new <= 0:
                break
            G = delete(G, min_desc[0], min_desc[1], min_desc[2])
            G = pdag2dag(G)
            G = dag2cpdag(G)
        else:
            score_new = score
            break

    return G
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 8,
        "end_line": 14,
        "content": _GES_FN,
    },
]
