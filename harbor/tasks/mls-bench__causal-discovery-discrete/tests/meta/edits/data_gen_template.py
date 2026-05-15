"""Data loader for bnlearn Bayesian network benchmarks.

Uses pgmpy's bundled BIF files (no network access needed) to load
real-world Bayesian networks, sample discrete observational data,
and extract the ground-truth DAG.
"""
import numpy as np
import pandas as pd

from causallearn.graph.Dag import Dag
from causallearn.graph.GraphNode import GraphNode

# All discrete bnlearn networks supported by pgmpy
SUPPORTED_NETWORKS = [
    "cancer", "earthquake", "survey", "asia", "sachs",
    "child", "insurance", "water", "mildew", "alarm",
    "barley", "hailfinder", "hepar2", "win95pts",
]


def load_and_sample(network_name, n_samples, seed):
    """Load a bnlearn network, sample data, and return (X, true_dag, node_names).

    Parameters
    ----------
    network_name : str
        One of the SUPPORTED_NETWORKS (e.g. 'asia', 'alarm', 'child').
    n_samples : int
        Number of observations to sample.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_variables), dtype int
        Integer-encoded discrete observational data.
    true_dag : causallearn.graph.Dag.Dag
        Ground-truth DAG as a causal-learn Dag object.
    node_names : list[str]
        Sorted variable names (consistent with column ordering in X).
    """
    from pgmpy.utils import get_example_model
    from pgmpy.sampling import BayesianModelSampling

    model = get_example_model(network_name)

    # Sample observational data
    sampler = BayesianModelSampling(model)
    df = sampler.forward_sample(size=n_samples, seed=seed)

    # Sorted node names for consistent ordering
    node_names = sorted(model.nodes())
    n_vars = len(node_names)

    # Integer-encode each variable (deterministic via sorted categories)
    X = np.zeros((n_samples, n_vars), dtype=int)
    for j, name in enumerate(node_names):
        col = df[name]
        categories = sorted(col.unique())
        cat_type = pd.CategoricalDtype(categories=categories, ordered=True)
        X[:, j] = col.astype(cat_type).cat.codes.values

    # Build true DAG as a causal-learn Dag object.
    # Use X1, X2, ... naming to match what causal-learn algorithms produce.
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    dag_nodes = [GraphNode(f"X{i + 1}") for i in range(n_vars)]
    true_dag = Dag(dag_nodes)
    for parent, child in model.edges():
        true_dag.add_directed_edge(dag_nodes[name_to_idx[parent]],
                                   dag_nodes[name_to_idx[child]])

    return X, true_dag, node_names
