"""PC baseline — replaces run_causal_discovery with PC using chi-squared test.

Reference: Spirtes et al., "Causation, Prediction, and Search", 2000.
Uses chi-squared conditional independence test for discrete data.
"""

_FILE = "causal-bnlearn/bench/custom_algorithm.py"

_PC_FN = """\
def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    \"\"\"
    Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
    \"\"\"
    from causallearn.utils.PCUtils import SkeletonDiscovery, Meek, UCSepset
    from causallearn.utils.cit import CIT

    alpha = 0.05
    indep_test = CIT(X, "chisq")

    # Step 1: skeleton discovery via chi-squared CI tests (stable PC)
    cg_1 = SkeletonDiscovery.skeleton_discovery(
        X, alpha, indep_test, stable=True,
        background_knowledge=None, verbose=False,
        show_progress=False, node_names=None,
    )

    # Step 2: orient unshielded colliders using UC-sepset rule (priority=2)
    cg_2 = UCSepset.uc_sepset(cg_1, 2, background_knowledge=None)

    # Step 3: complete orientation with Meek rules
    cg = Meek.meek(cg_2, background_knowledge=None)

    return cg.G
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 8,
        "end_line": 14,
        "content": _PC_FN,
    },
]
