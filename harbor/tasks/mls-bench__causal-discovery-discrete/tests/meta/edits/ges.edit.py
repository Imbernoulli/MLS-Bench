"""GES baseline — replaces run_causal_discovery with GES using BDeu score.

Reference: Chickering, "Optimal structure identification with greedy search", 2002.
Uses BDeu (Bayesian Dirichlet equivalent uniform) score for discrete data.
"""

_FILE = "causal-bnlearn/bench/custom_algorithm.py"

_GES_FN = """\
def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    \"\"\"
    Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
    \"\"\"
    from causallearn.search.ScoreBased.GES import ges

    result = ges(
        X,
        score_func="local_score_BDeu",
        parameters={"sample_prior": 1.0, "structure_prior": 1.0},
    )
    return result["G"]
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
