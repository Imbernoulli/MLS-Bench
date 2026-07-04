"""Evaluation metrics for CPDAG recovery on linear Gaussian data."""
from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from causallearn.graph.ArrowConfusion import ArrowConfusion
from causallearn.graph.Graph import Graph
from causallearn.graph.SHD import SHD
from causallearn.utils.DAG2CPDAG import dag2cpdag


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator > 0 else 0.0


def _normalize_graph_output(graph_output):
    """Normalize algorithm output to a causallearn Graph object."""
    if isinstance(graph_output, dict) and "G" in graph_output:
        graph_output = graph_output["G"]
    elif hasattr(graph_output, "G"):
        graph_output = graph_output.G

    if not isinstance(graph_output, Graph):
        raise TypeError(
            "run_causal_discovery must return a causallearn Graph/GeneralGraph "
            f"(or wrapper with .G / dict['G']); got {type(graph_output)!r}"
        )
    return graph_output


def compute_metrics(est_graph, true_dag):
    """Compute SHD and precision/recall metrics on CPDAGs."""
    est_cpdag = _normalize_graph_output(est_graph)
    true_cpdag = dag2cpdag(true_dag)

    shd = SHD(true_cpdag, est_cpdag).get_shd()

    adj = AdjacencyConfusion(true_cpdag, est_cpdag)
    adj_tp = adj.get_adj_tp()
    adj_fp = adj.get_adj_fp()
    adj_fn = adj.get_adj_fn()
    adj_precision = _safe_div(adj_tp, adj_tp + adj_fp)
    adj_recall = _safe_div(adj_tp, adj_tp + adj_fn)

    arrow = ArrowConfusion(true_cpdag, est_cpdag)
    arrow_tp = arrow.get_arrows_tp()
    arrow_fp = arrow.get_arrows_fp()
    arrow_fn = arrow.get_arrows_fn()
    arrow_precision = _safe_div(arrow_tp, arrow_tp + arrow_fp)
    arrow_recall = _safe_div(arrow_tp, arrow_tp + arrow_fn)

    return {
        "shd": int(shd),
        "adj_precision": float(adj_precision),
        "adj_recall": float(adj_recall),
        "arrow_precision": float(arrow_precision),
        "arrow_recall": float(arrow_recall),
    }
