"""Synthetic nonlinear additive noise model (ANM) data generator.

Structural equation:  x_j = f_j(pa(j)) + e_j
where f_j is a nonlinear function and e_j is additive non-Gaussian noise.
"""
import numpy as np
import networkx as nx


def simulate_dag(n_nodes, graph_type, seed, er_prob=0.3, sf_m=2):
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


def _random_nonlinear_fn(rng, fn_type="mixed"):
    """Return a random nonlinear scalar function R -> R.

    fn_type:
      'gp'    — random GP sample (smooth nonlinear)
      'mlp'   — random 1-hidden-layer MLP
      'mixed' — randomly pick from {gp, mlp, polynomial, sigmoid-combo}
    """
    if fn_type == "mixed":
        choice = rng.choice(["gp", "mlp", "poly", "sigmoid"])
    else:
        choice = fn_type

    if choice == "gp":
        # Approximate a GP sample via random Fourier features
        n_features = 100
        length_scale = rng.uniform(0.5, 2.0)
        W = rng.standard_normal(n_features) / length_scale
        b = rng.uniform(0, 2 * np.pi, n_features)
        alpha = rng.standard_normal(n_features) * np.sqrt(2.0 / n_features)

        def fn(x):
            # x is 1D array
            features = np.cos(np.outer(x, W) + b)  # (n, n_features)
            return features @ alpha

        return fn

    elif choice == "mlp":
        hidden_dim = rng.integers(10, 30)
        W1 = rng.standard_normal(hidden_dim)
        b1 = rng.standard_normal(hidden_dim) * 0.5
        W2 = rng.standard_normal(hidden_dim)

        def fn(x):
            h = np.tanh(np.outer(x, W1) + b1)  # (n, hidden_dim)
            return h @ W2

        return fn

    elif choice == "poly":
        degree = rng.integers(2, 4)
        coeffs = rng.standard_normal(degree + 1) * 0.5
        coeffs[0] = 0  # no constant term

        def fn(x):
            result = np.zeros_like(x, dtype=float)
            for d in range(1, degree + 1):
                result += coeffs[d] * (x ** d)
            return result

        return fn

    else:  # sigmoid
        # Use steeper sigmoid (a in [3, 8]) to ensure strong nonlinearity.
        # With a in [0.5, 2.0] the sigmoid is nearly linear over typical
        # data ranges, allowing linear methods like DirectLiNGAM to succeed.
        a = rng.uniform(3.0, 8.0) * rng.choice([-1, 1])
        b = rng.uniform(-1, 1)
        c = rng.uniform(1.0, 3.0) * rng.choice([-1, 1])

        def fn(x):
            return c / (1 + np.exp(-a * (x - b)))

        return fn


def simulate_nonlinear_anm(n_nodes, n_samples, graph_type, noise_type, seed,
                           er_prob=0.3, sf_m=2, fn_type="mixed"):
    """Generate observational data from a nonlinear additive noise model (ANM).

    Structural equation:  x_j = f_j(parents(j)) + e_j
    where f_j is a nonlinear function of all parents (summed contributions),
    and e_j is additive noise.

    Returns
    -------
    X : ndarray, shape (n_samples, n_nodes)
        Observed data matrix.
    B_true : ndarray, shape (n_nodes, n_nodes)
        Ground-truth binary adjacency matrix.  B_true[i, j] = 1 means j -> i.
    """
    rng = np.random.default_rng(seed)

    # --- DAG structure: struct[i, j] = 1 means i -> j ---
    struct = simulate_dag(n_nodes, graph_type, seed=seed, er_prob=er_prob, sf_m=sf_m)

    # Convert to B_true convention: B_true[child, parent] = 1 means parent -> child
    B_true = struct.T.copy()

    # --- Generate nonlinear functions for each parent-child pair ---
    edge_fns = {}
    for c in range(n_nodes):
        parents = np.where(struct[:, c] == 1)[0]
        for p in parents:
            edge_fns[(p, c)] = _random_nonlinear_fn(rng, fn_type)

    # --- Noise ---
    if noise_type == "exp":
        noise = rng.exponential(scale=1.0, size=(n_samples, n_nodes))
        noise -= noise.mean(axis=0)
    elif noise_type == "laplace":
        noise = rng.laplace(loc=0.0, scale=1.0, size=(n_samples, n_nodes))
    elif noise_type == "uniform":
        noise = rng.uniform(-np.sqrt(3), np.sqrt(3), size=(n_samples, n_nodes))
    elif noise_type == "gaussian":
        noise = rng.standard_normal(size=(n_samples, n_nodes))
    else:
        raise ValueError(
            f"Unknown noise_type: {noise_type!r}. "
            "Choose 'exp', 'laplace', 'uniform', or 'gaussian'."
        )

    # --- Generate data in topological order ---
    X = np.zeros((n_samples, n_nodes))
    # Topological order: since struct enforces i < j for edges, node 0, 1, ... is valid
    for j in range(n_nodes):
        parents = np.where(struct[:, j] == 1)[0]
        X[:, j] = noise[:, j]
        for p in parents:
            X[:, j] += edge_fns[(p, j)](X[:, p])

    return X, B_true
