"""BOSS baseline — replaces run_causal_discovery with BOSS using BDeu score.

Reference: Andrews et al., "Fast Scalable and Accurate Discovery of DAGs
Using the Best Order Score Search and Grow-Shrink Trees", NeurIPS 2023.
Uses BDeu score for discrete data with variable mutation operators.
"""

_FILE = "causal-bnlearn/bench/custom_algorithm.py"

_BOSS_FN = """\
def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    \"\"\"
    Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
    \"\"\"
    from causallearn.search.PermutationBased.BOSS import boss

    G = boss(
        X,
        score_func="local_score_BDeu",
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
        "content": _BOSS_FN,
    },
]
