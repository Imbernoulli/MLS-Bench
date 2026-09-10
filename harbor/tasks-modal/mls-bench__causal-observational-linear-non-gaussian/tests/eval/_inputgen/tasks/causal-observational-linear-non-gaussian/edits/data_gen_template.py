"""Synthetic linear non-Gaussian DAG data generator for LiNGAM benchmarking."""
import numpy as np
import networkx as nx


def simulate_dag(n_nodes, graph_type, seed, er_prob=0.5, sf_m=2):
    """Return a binary adjacency matrix for a random DAG.

    Convention: adj[i, j] = 1 means i -> j  (i is a parent of j).
    The DAG is enforced by keeping only edges i -> j with i < j, imposing
    a topological ordering by node index.
    """
    rng = np.random.default_rng(seed)
    graph_seed = int(rng.integers(0, 2**31 - 1))

    if graph_type == "er":
        G = nx.erdos_renyi_graph(n_nodes, er_prob, seed=graph_seed, directed=True)
        adj = nx.to_numpy_array(G)
        adj = np.triu(adj, k=1)  # enforce DAG: keep only i < j directed edges
    elif graph_type == "sf":
        # Barabasi-Albert model; convert undirected to DAG by node index order
        G = nx.barabasi_albert_graph(n_nodes, sf_m, seed=graph_seed)
        adj = np.zeros((n_nodes, n_nodes))
        for u, v in G.edges():
            lo, hi = min(u, v), max(u, v)
            adj[lo, hi] = 1  # enforce DAG: lower-index node is the parent
    else:
        raise ValueError(f"Unknown graph_type: {graph_type!r}. Choose 'er' or 'sf'.")

    return adj


def simulate_lingam(n_nodes, n_samples, graph_type, noise_type, seed,
                    er_prob=0.5, sf_m=2, weight_low=0.5, weight_high=2.0):
    """Generate observational data from a linear non-Gaussian DAG (LiNGAM model).

    Structural equation:  x_i = sum_{j: j->i} B[i,j] * x_j + e_i
    In matrix form:  (I - B) X^T = E^T,  solved as X^T = (I - B)^{-1} E^T.

    Returns
    -------
    X : ndarray, shape (n_samples, n_nodes)
        Observed data matrix.
    B_true : ndarray, shape (n_nodes, n_nodes)
        Ground-truth adjacency matrix.  B_true[i, j] != 0 means j -> i.
    """
    rng = np.random.default_rng(seed)

    # --- DAG structure: struct[i, j] = 1 means i -> j ---------------------------
    struct = simulate_dag(n_nodes, graph_type, seed=seed, er_prob=er_prob, sf_m=sf_m)

    # --- Edge weights sampled from [-weight_high, -weight_low] ∪ [weight_low, weight_high]
    raw_weights = rng.uniform(weight_low, weight_high, size=(n_nodes, n_nodes))
    signs = rng.choice([-1, 1], size=(n_nodes, n_nodes))
    raw_weights = raw_weights * signs

    # B[child, parent] = weight  (causal-learn convention: B[i,j] means j->i)
    # struct[p, c] = 1 (edge p->c) => B[c, p] = raw_weights[p, c]
    B_true = np.zeros((n_nodes, n_nodes))
    parents, children = np.where(struct == 1)
    for p, c in zip(parents, children):
        B_true[c, p] = raw_weights[p, c]

    # --- Noise ---------------------------------------------------------------
    if noise_type == "exp":
        noise = rng.exponential(scale=1.0, size=(n_samples, n_nodes))
        noise -= noise.mean(axis=0)  # center so E[e_i] = 0
    elif noise_type == "laplace":
        noise = rng.laplace(loc=0.0, scale=1.0, size=(n_samples, n_nodes))
    elif noise_type == "uniform":
        noise = rng.uniform(-np.sqrt(3), np.sqrt(3), size=(n_samples, n_nodes))
    else:
        raise ValueError(
            f"Unknown noise_type: {noise_type!r}. Choose 'exp', 'laplace', or 'uniform'."
        )

    # --- Solve: X^T = (I - B)^{-1} E^T  -------------------------------------
    I = np.eye(n_nodes)
    X = np.linalg.solve(I - B_true, noise.T).T  # (n_samples, n_nodes)

    return X, B_true
