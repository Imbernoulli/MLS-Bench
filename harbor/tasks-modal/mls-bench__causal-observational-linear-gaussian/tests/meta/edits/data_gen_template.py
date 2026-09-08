"""Synthetic linear Gaussian DAG data generator for CPDAG benchmarking."""
import networkx as nx
import numpy as np

from causallearn.graph.Dag import Dag
from causallearn.graph.GraphNode import GraphNode


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
