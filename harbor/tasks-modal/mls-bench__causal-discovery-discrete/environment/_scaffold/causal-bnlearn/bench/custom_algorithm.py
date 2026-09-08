import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

# =====================================================================
# EDITABLE: implement run_causal_discovery below
# =====================================================================
def run_causal_discovery(X: np.ndarray) -> GeneralGraph:
    """
    Input:  X of shape (n_samples, n_variables), integer-encoded discrete data
    Output: estimated CPDAG as causallearn.graph.GeneralGraph.GeneralGraph
    """
    nodes = [GraphNode(f"X{i + 1}") for i in range(X.shape[1])]
    return GeneralGraph(nodes)
# =====================================================================
