"""GRaSP baseline — replaces run_causal_discovery with GRaSP using BDeu score.

Reference: Lam et al., "Greedy Relaxations of the Sparsest Permutation Algorithm", 2022.
Uses BDeu score for discrete data with depth-3 DFS relaxation.
"""

_FILE = "causal-bnlearn/bench/custom_algorithm.py"

_GRASP_FN = """\
def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    \"\"\"
    Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
    \"\"\"
    from causallearn.search.PermutationBased.GRaSP import grasp

    G = grasp(
        X,
        score_func="local_score_BDeu",
        depth=3,
        parameters={"sample_prior": 1.0, "structure_prior": 1.0},
    )
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
