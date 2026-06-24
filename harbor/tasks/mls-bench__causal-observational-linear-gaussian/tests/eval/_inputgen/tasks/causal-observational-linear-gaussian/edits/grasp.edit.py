"""GRaSP baseline — replaces run_causal_discovery with GRaSP.

Reference: Lam et al., "Greedy Relaxations of the Sparsest Permutation Algorithm", 2022.
"""

_FILE = "causal-learn/bench/custom_algorithm.py"

_GRASP_FN = """\
class _GraspOrder:
    def __init__(self, p, score):
        self.order = list(range(p))
        self.parents = {}
        self.local_scores = {}
        self.edges = 0
        import random
        random.shuffle(self.order)
        for i in range(p):
            y = self.order[i]
            self.parents[y] = []
            self.local_scores[y] = -score.score(y, [])

    def get(self, i): return self.order[i]
    def set(self, i, y): self.order[i] = y
    def index(self, y): return self.order.index(y)
    def insert(self, i, y): self.order.insert(i, y)
    def pop(self, i=-1): return self.order.pop(i)
    def get_parents(self, y): return self.parents[y]
    def set_parents(self, y, yp): self.parents[y] = yp
    def get_local_score(self, y): return self.local_scores[y]
    def set_local_score(self, y, s): self.local_scores[y] = s
    def get_edges(self): return self.edges
    def set_edges(self, e): self.edges = e
    def bump_edges(self, b): self.edges += b
    def len(self): return len(self.order)


def _grasp_get_ancestors(y, ancestors, order):
    ancestors.append(y)
    for x in order.get_parents(y):
        if x not in ancestors:
            _grasp_get_ancestors(x, ancestors, order)


def _grasp_tuck(i, j, order):
    ancestors = []
    _grasp_get_ancestors(order.get(i), ancestors, order)
    shift = 0
    for k in range(j + 1, i + 1):
        if order.get(k) in ancestors:
            order.insert(j + shift, order.pop(k))
            shift += 1


def _grasp_update(i, j, order, gsts):
    edge_bump = 0
    old_score = 0
    new_score = 0
    for k in range(j, i + 1):
        z = order.get(k)
        z_parents = order.get_parents(z)
        edge_bump -= len(z_parents)
        old_score += order.get_local_score(z)
        z_parents.clear()
        candidates = [order.get(l) for l in range(0, k)]
        local_score = gsts[z].trace(candidates, z_parents)
        order.set_local_score(z, local_score)
        edge_bump += len(z_parents)
        new_score += local_score
    return edge_bump, new_score - old_score


def _grasp_dfs(depth, flipped, history, order, gsts):
    import random
    cache = [{}, {}, {}, 0]
    indices = list(range(order.len()))
    random.shuffle(indices)
    for i in indices:
        y = order.get(i)
        y_parents = order.get_parents(y)
        random.shuffle(y_parents)
        for x in y_parents:
            covered = set([x] + order.get_parents(x)) == set(y_parents)
            if len(history) > 0 and not covered:
                continue
            j = order.index(x)
            for k in range(j, i + 1):
                z = order.get(k)
                cache[0][k] = z
                cache[1][k] = order.get_parents(z)[:]
                cache[2][k] = order.get_local_score(z)
            cache[3] = order.get_edges()
            _grasp_tuck(i, j, order)
            edge_bump, score_bump = _grasp_update(i, j, order, gsts)
            if score_bump > 1e-6:
                order.bump_edges(edge_bump)
                return True
            if score_bump > -1e-6:
                flipped = flipped ^ set(
                    [
                        tuple(sorted([x, z]))
                        for z in order.get_parents(x)
                        if order.index(z) < i
                    ]
                )
                if len(flipped) > 0 and flipped not in history:
                    history.append(flipped)
                    if depth > 0 and _grasp_dfs(depth - 1, flipped, history, order, gsts):
                        return True
                    del history[-1]
            for k in range(j, i + 1):
                z = cache[0][k]
                order.set(k, z)
                order.set_parents(z, cache[1][k])
                order.set_local_score(z, cache[2][k])
            order.set_edges(cache[3])
    return False


def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    \"\"\"
    Input:  X of shape (n_samples, n_variables)
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
    \"\"\"
    from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
    from causallearn.score.LocalScoreFunction import local_score_BIC_from_cov
    from causallearn.search.PermutationBased.gst import GST
    from causallearn.utils.DAG2CPDAG import dag2cpdag

    X = X.copy()
    n, p = X.shape
    depth = 3
    parameters = {"lambda_value": 2}
    score = LocalScoreClass(
        data=X, local_score_fun=local_score_BIC_from_cov, parameters=parameters
    )
    gsts = [GST(i, score) for i in range(p)]

    nodes = [GraphNode("X%d" % (i + 1)) for i in range(p)]
    G = GeneralGraph(nodes)

    order = _GraspOrder(p, score)

    for i in range(p):
        y = order.get(i)
        y_parents = order.get_parents(y)
        candidates = [order.get(j) for j in range(0, i)]
        local_score = gsts[y].trace(candidates, y_parents)
        order.set_local_score(y, local_score)
        order.bump_edges(len(y_parents))

    while _grasp_dfs(depth - 1, set(), [], order, gsts):
        pass

    for y in range(p):
        for x in order.get_parents(y):
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
        "content": _GRASP_FN,
    },
]
