"""BOSS baseline — replaces run_causal_discovery with BOSS.

Reference: Andrews et al., "Fast Scalable and Accurate Discovery of DAGs
Using the Best Order Score Search and Grow-Shrink Trees", NeurIPS 2023.
"""

_FILE = "causal-learn/bench/custom_algorithm.py"

_BOSS_FN = """\
def _boss_reversed_enumerate(iter_, j):
    for w in reversed(iter_):
        yield j, w
        j -= 1


def _boss_better_mutation(v, order, gsts):
    i = order.index(v)
    p = len(order)
    scores = np.zeros(p + 1)

    prefix = []
    score = 0
    for j, w in enumerate(order):
        scores[j] = gsts[v].trace(prefix) + score
        if v != w:
            score += gsts[w].trace(prefix)
            prefix.append(w)

    scores[p] = gsts[v].trace(prefix) + score
    best = p

    prefix.append(v)
    score = 0
    for j, w in _boss_reversed_enumerate(order, p - 1):
        if v != w:
            prefix.remove(w)
            score += gsts[w].trace(prefix)
        scores[j] += score
        if scores[j] > scores[best]:
            best = j

    if scores[i] + 1e-6 > scores[best]:
        return False
    order.remove(v)
    order.insert(best - int(best > i), v)
    return True


def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    \"\"\"
    Input:  X of shape (n_samples, n_variables)
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
    \"\"\"
    import random
    import sys
    import time
    from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
    from causallearn.score.LocalScoreFunction import local_score_BIC_from_cov
    from causallearn.search.PermutationBased.gst import GST
    from causallearn.utils.DAG2CPDAG import dag2cpdag

    X = X.copy()
    n, p = X.shape
    parameters = {"lambda_value": 2}
    score = LocalScoreClass(
        data=X, local_score_fun=local_score_BIC_from_cov, parameters=parameters
    )

    nodes = [GraphNode("X%d" % (i + 1)) for i in range(p)]
    G = GeneralGraph(nodes)

    order = list(range(p))
    gsts = [GST(v, score) for v in order]
    parents = {v: [] for v in order}

    variables = list(order)
    while True:
        improved = False
        random.shuffle(variables)
        for v in variables:
            improved |= _boss_better_mutation(v, order, gsts)
        if not improved:
            break

    for i, v in enumerate(order):
        parents[v].clear()
        gsts[v].trace(order[:i], parents[v])

    for y in range(p):
        for x in parents[y]:
            G.add_directed_edge(nodes[x], nodes[y])

    G = dag2cpdag(G)
    return G
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 8,
        "end_line": 14,
        "content": _BOSS_FN,
    },
]
