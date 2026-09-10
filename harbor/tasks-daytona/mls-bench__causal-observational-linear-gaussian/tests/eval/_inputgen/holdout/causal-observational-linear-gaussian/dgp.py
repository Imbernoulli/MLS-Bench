"""Held-out DGP + scoring module for causal-observational-linear-gaussian.

This module lives OUTSIDE every path that is bind-mounted into the agent's
container (workspace package dir, task dir, data binds). It is imported only by
the host-side ``parser.py`` and the input pre-generator in ``mid_edit.py`` —
never by the agent-editable ``custom_algorithm.py`` or the FIXED ``run_eval.py``
that runs inside the container. It holds the data-generating process (which also
produces the true DAG) and the CPDAG metrics, so the agent's process can never
reach the answer.

The DGP and metric bodies are byte-identical to the originals that used to live
in ``edits/data_gen_template.py`` and ``edits/metrics_template.py`` — so the
data, and therefore every honest result, is reproduced exactly.
"""

import networkx as nx
import numpy as np

from causallearn.graph.Dag import Dag
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from causallearn.graph.ArrowConfusion import ArrowConfusion
from causallearn.graph.Graph import Graph
from causallearn.graph.SHD import SHD
from causallearn.utils.DAG2CPDAG import dag2cpdag


# =====================================================================
# Data Generating Process (byte-identical to data_gen_template.py)
# =====================================================================

def simulate_dag(n_nodes, graph_type, seed, er_prob=0.5, sf_m=2):
    """Return a binary adjacency matrix with convention adj[i, j] = 1 for i -> j."""
    rng = np.random.default_rng(seed)
    graph_seed = int(rng.integers(0, 2**31 - 1))

    if graph_type == "er":
        graph = nx.erdos_renyi_graph(n_nodes, er_prob, seed=graph_seed, directed=True)
        adj = nx.to_numpy_array(graph)
        adj = np.triu(adj, k=1)
    elif graph_type == "sf":
        graph = nx.barabasi_albert_graph(n_nodes, sf_m, seed=graph_seed)
        adj = np.zeros((n_nodes, n_nodes))
        for u, v in graph.edges():
            lo, hi = min(u, v), max(u, v)
            adj[lo, hi] = 1
    else:
        raise ValueError(f"Unknown graph_type: {graph_type!r}. Choose 'er' or 'sf'.")

    return adj


def _dag_from_structure(struct):
    """Build a causallearn Dag object from parent->child binary structure."""
    n_nodes = struct.shape[0]
    nodes = [GraphNode(f"X{i + 1}") for i in range(n_nodes)]
    dag = Dag(nodes)

    parents, children = np.where(struct == 1)
    for p, c in zip(parents, children):
        dag.add_directed_edge(nodes[p], nodes[c])
    return dag


def simulate_linear_gaussian(
    n_nodes,
    n_samples,
    graph_type,
    seed,
    er_prob=0.5,
    sf_m=2,
    weight_low=0.2,
    weight_high=1.5,
    noise_scale=1.0,
):
    """Generate observational data from a linear Gaussian SEM and the true Dag."""
    rng = np.random.default_rng(seed)
    struct = simulate_dag(n_nodes, graph_type, seed=seed, er_prob=er_prob, sf_m=sf_m)

    raw_weights = rng.uniform(weight_low, weight_high, size=(n_nodes, n_nodes))
    signs = rng.choice([-1.0, 1.0], size=(n_nodes, n_nodes))
    raw_weights = raw_weights * signs

    # B[child, parent] = weight in causal-learn convention.
    B_true = np.zeros((n_nodes, n_nodes))
    parents, children = np.where(struct == 1)
    for p, c in zip(parents, children):
        B_true[c, p] = raw_weights[p, c]

    noise = rng.normal(loc=0.0, scale=noise_scale, size=(n_samples, n_nodes))
    X = np.linalg.solve(np.eye(n_nodes) - B_true, noise.T).T

    true_dag = _dag_from_structure(struct)
    return X, true_dag


# =====================================================================
# Metrics (byte-identical to metrics_template.py)
# =====================================================================

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


# =====================================================================
# Helpers for the input pre-generator (mid_edit) and host-side scorer (parser)
# =====================================================================

def _args_from_key(graph_type, n_nodes, n_samples, er_prob, sf_m, noise_scale, seed):
    """Normalize CLI-style args (parser/mid_edit pass the same set)."""
    return dict(
        n_nodes=int(n_nodes),
        n_samples=int(n_samples),
        graph_type=str(graph_type),
        seed=int(seed),
        er_prob=float(er_prob),
        sf_m=int(sf_m),
        noise_scale=float(noise_scale),
    )


def gen_input(graph_type, n_nodes, n_samples, er_prob, sf_m, noise_scale, seed):
    """Return ONLY the agent-visible observational data X for a given setting.

    The true DAG is computed internally by the DGP but deliberately NOT returned
    here, so the pre-generator that writes the agent's input file never persists
    the answer.
    """
    a = _args_from_key(graph_type, n_nodes, n_samples, er_prob, sf_m, noise_scale, seed)
    X, _true_dag = simulate_linear_gaussian(**a)
    return X


def truth(graph_type, n_nodes, n_samples, er_prob, sf_m, noise_scale, seed):
    """Return the held-out ground-truth Dag for the host-side scorer."""
    a = _args_from_key(graph_type, n_nodes, n_samples, er_prob, sf_m, noise_scale, seed)
    _X, true_dag = simulate_linear_gaussian(**a)
    return true_dag
