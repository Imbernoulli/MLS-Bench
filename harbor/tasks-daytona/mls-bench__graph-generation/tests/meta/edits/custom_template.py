"""Graph Generation Benchmark.

Train a generative model on small graph datasets and evaluate using MMD statistics.

FIXED: Dataset loading/generation, graph statistics computation, MMD evaluation,
       training loop orchestration, argument parsing.
EDITABLE: GraphGenerator class (the generative model).

Usage:
    python pytorch-geometric/custom_graphgen.py --dataset community_small --seed 42
"""

import argparse
import math
import os
import random
import warnings
from collections import defaultdict
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# ============================================================================
# Dataset Generation & Loading (FIXED)
# ============================================================================

def generate_community_small(n_graphs=100, min_nodes=12, max_nodes=20):
    """Generate community_small dataset: 2-community graphs.

    Each graph has 2 communities connected by a few inter-community edges.
    Uses the small-community setup common in GraphRNN/GDSS-style benchmarks.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required: pip install networkx")

    graphs = []
    for _ in range(n_graphs):
        n = random.randint(min_nodes, max_nodes)
        n1 = n // 2
        n2 = n - n1
        G = nx.planted_partition_graph(2, n // 2, p_in=0.7, p_out=0.05)
        G = G.to_undirected()
        G.remove_edges_from(nx.selfloop_edges(G))
        graphs.append(G)
    return graphs


def generate_ego_small(n_max=200):
    """Generate ego_small dataset: small ego graphs from Citeseer.

    Uses PyG's Planetoid(Citeseer) and extracts ego graphs of 1-hop neighborhoods.
    Returns up to n_max graphs with 4-18 nodes.
    """
    try:
        import networkx as nx
        from torch_geometric.datasets import Planetoid
        from torch_geometric.utils import to_networkx
    except ImportError:
        raise ImportError("torch_geometric and networkx required")

    dataset = Planetoid(root=os.environ.get("DATA_ROOT", "/data") + "/Planetoid", name="CiteSeer")
    data = dataset[0]
    G_full = to_networkx(data, to_undirected=True)

    graphs = []
    nodes = list(G_full.nodes())
    random.shuffle(nodes)
    for node in nodes:
        ego = nx.ego_graph(G_full, node, radius=1)
        if 4 <= ego.number_of_nodes() <= 18:
            # Relabel to consecutive integers
            ego = nx.convert_node_labels_to_integers(ego)
            ego.remove_edges_from(nx.selfloop_edges(ego))
            graphs.append(ego)
        if len(graphs) >= n_max:
            break
    return graphs


def load_enzymes(n_max=587):
    """Load ENZYMES dataset from TUDataset.

    Protein tertiary structure graphs, 587 graphs, 10-125 nodes.
    """
    try:
        import networkx as nx
        from torch_geometric.datasets import TUDataset
        from torch_geometric.utils import to_networkx
    except ImportError:
        raise ImportError("torch_geometric and networkx required")

    dataset = TUDataset(root=os.environ.get("DATA_ROOT", "/data") + "/TUDataset", name="ENZYMES")
    graphs = []
    for i in range(min(len(dataset), n_max)):
        data = dataset[i]
        G = to_networkx(data, to_undirected=True)
        G = nx.convert_node_labels_to_integers(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        if G.number_of_nodes() >= 2:
            graphs.append(G)
    return graphs


def load_dataset(name: str):
    """Load a graph dataset by name. Returns list of networkx graphs."""
    if name == "community_small":
        return generate_community_small()
    elif name == "ego_small":
        return generate_ego_small()
    elif name == "enzymes":
        return load_enzymes()
    else:
        raise ValueError(f"Unknown dataset: {name}")


# ============================================================================
# Graph Representation Utilities (FIXED)
# ============================================================================

def graphs_to_adj(graphs, max_nodes=None):
    """Convert networkx graphs to padded adjacency matrices.

    Returns:
        adjs: Tensor [N, max_nodes, max_nodes] (binary adjacency)
        node_counts: Tensor [N] (actual number of nodes per graph)
    """
    if max_nodes is None:
        max_nodes = max(G.number_of_nodes() for G in graphs)
    adjs = []
    node_counts = []
    for G in graphs:
        n = G.number_of_nodes()
        A = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        for u, v in G.edges():
            if u < max_nodes and v < max_nodes:
                A[u, v] = 1.0
                A[v, u] = 1.0
        adjs.append(A)
        node_counts.append(n)
    return torch.tensor(np.array(adjs)), torch.tensor(node_counts, dtype=torch.long)


def adj_to_graphs(adjs, node_counts):
    """Convert adjacency matrices back to networkx graphs.

    Args:
        adjs: Tensor or ndarray [N, max_nodes, max_nodes]
        node_counts: Tensor or list [N]

    Returns:
        List of networkx graphs.
    """
    import networkx as nx
    if isinstance(adjs, torch.Tensor):
        adjs = adjs.detach().cpu().numpy()
    if isinstance(node_counts, torch.Tensor):
        node_counts = node_counts.cpu().tolist()

    graphs = []
    for A, n in zip(adjs, node_counts):
        n = int(n)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        A_sub = A[:n, :n]
        # Threshold at 0.5 for probabilistic outputs
        A_bin = (A_sub > 0.5).astype(int)
        # Make symmetric and remove self-loops
        A_sym = np.maximum(A_bin, A_bin.T)
        np.fill_diagonal(A_sym, 0)
        for i in range(n):
            for j in range(i + 1, n):
                if A_sym[i, j]:
                    G.add_edge(i, j)
        graphs.append(G)
    return graphs


class GraphDataset(Dataset):
    """Dataset of adjacency matrices for graph generation."""

    def __init__(self, adjs, node_counts):
        self.adjs = adjs        # [N, max_nodes, max_nodes]
        self.node_counts = node_counts  # [N]

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        return self.adjs[idx], self.node_counts[idx]


# ============================================================================
# Graph Statistics & MMD Evaluation (FIXED)
# ============================================================================

def degree_stats(G):
    """Degree sequence of graph (sorted)."""
    return sorted([d for _, d in G.degree()], reverse=True)


def clustering_stats(G):
    """Clustering coefficient distribution."""
    import networkx as nx
    cc = nx.clustering(G)
    return sorted(cc.values(), reverse=True)


def orbit_stats(G):
    """4-orbit count distribution.

    Returns an approximate per-graph orbit feature vector in a GDSS-style format:
    a single feature vector per graph (mean orbit count per node for several
    orbit types). Unlike degree/clustering, the orbit MMD in GDSS is computed
    directly on these raw per-graph vectors with a Gaussian kernel (is_hist=False),
    NOT histogrammed. Returning a numpy vector here signals this to compute_mmd.
    """
    import networkx as nx
    n = G.number_of_nodes()
    # 4 orbit-like counts per node: triangles, 2-paths, 3-paths, 4-cliques-participation
    if n < 1:
        return np.zeros(4, dtype=np.float64)

    triangles = np.zeros(n, dtype=np.float64)
    two_paths = np.zeros(n, dtype=np.float64)
    three_paths = np.zeros(n, dtype=np.float64)
    four_cliques = np.zeros(n, dtype=np.float64)

    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    adj = {v: set(G.neighbors(v)) for v in nodes}

    for v in nodes:
        Nv = adj[v]
        # triangles through v
        nbrs = list(Nv)
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                if nbrs[j] in adj[nbrs[i]]:
                    triangles[idx[v]] += 1
        # 2-paths (v -- u -- w, w != v, w not adj to v)
        for u in Nv:
            for w in adj[u]:
                if w != v and w not in Nv:
                    two_paths[idx[v]] += 1
        # 3-paths starting at v (v-u-w-x, distinct, no shortcut to v)
        for u in Nv:
            for w in adj[u]:
                if w == v or w in Nv:
                    continue
                for x in adj[w]:
                    if x != v and x != u and x not in Nv:
                        three_paths[idx[v]] += 1
        # 4-clique participation
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                if nbrs[j] not in adj[nbrs[i]]:
                    continue
                for k in range(j + 1, len(nbrs)):
                    if nbrs[k] in adj[nbrs[i]] and nbrs[k] in adj[nbrs[j]]:
                        four_cliques[idx[v]] += 1

    # Per-graph feature: mean count across nodes for each orbit type
    vec = np.array([
        triangles.mean(),
        two_paths.mean(),
        three_paths.mean(),
        four_cliques.mean(),
    ], dtype=np.float64)
    return vec


def _to_histogram(values, n_bins=100):
    """Convert a list of values to a normalized histogram (probability distribution).

    This is the standard approach used in graph generation benchmarks (GraphRNN,
    GRAN, GDSS) to make statistics comparable across graphs of different sizes.
    """
    values = np.array(values, dtype=np.float64)
    if len(values) == 0:
        return np.zeros(n_bins)
    # Use fixed bin edges across the full range
    hist, _ = np.histogram(values, bins=n_bins, range=(values.min(), max(values.max(), values.min() + 1e-8)), density=False)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist.astype(np.float64)


def _gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
    """Gaussian kernel with an Earth Mover's Distance inner metric.

    GDSS-style Gaussian kernel over a 1-D EMD distance between probability
    histograms. Falls back to simple L1 when scipy is unavailable.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    # Try scipy's Wasserstein (EMD) on 1-D distributions
    try:
        from scipy.stats import wasserstein_distance
        support = np.arange(len(x)) * distance_scaling
        d = wasserstein_distance(support, support, u_weights=x, v_weights=y)
    except Exception:
        # Fallback: L1 / total variation times distance_scaling as a proxy
        d = 0.5 * np.sum(np.abs(x - y)) * distance_scaling
    return float(np.exp(-(d * d) / (2.0 * sigma * sigma)))


def _gaussian(x, y, sigma=1.0):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    d2 = float(np.sum((x - y) ** 2))
    return float(np.exp(-d2 / (2.0 * sigma * sigma)))


def compute_mmd(samples1, samples2, kernel="gaussian_emd", sigma=1.0, is_hist=True):
    """Compute MMD between two sets of graph statistics (biased estimator).

    Uses GDSS-style graph-statistic MMD conventions:
      - For degree/clustering: each sample is a 1-D list of per-node values
        which is converted to a normalized histogram, then MMD uses a Gaussian-EMD
        kernel with sigma=1.0.
      - For orbit: each sample is already a per-graph feature vector (is_hist=False),
        MMD uses a plain Gaussian kernel with sigma=30.0.
    """
    # Pick kernel
    if kernel == "gaussian_emd":
        kfn = lambda a, b: _gaussian_emd(a, b, sigma=sigma)
    else:
        kfn = lambda a, b: _gaussian(a, b, sigma=sigma)

    if is_hist:
        # Build histograms with integer bins over the global value range.
        # Degree/clustering: degrees are integers; clustering is in [0,1].
        # Use fixed bins: for degree, bins = max_degree+1; for clustering,
        # bins = 100 over [0,1]. We pick bins based on value type.
        all_vals = [v for s in list(samples1) + list(samples2) for v in s]
        if len(all_vals) == 0:
            return 0.0
        vmin = float(np.min(all_vals))
        vmax = float(np.max(all_vals))
        if vmax <= 1.0 and vmin >= 0.0:
            # Clustering coefficient in [0, 1]
            n_bins = 100
            bin_range = (0.0, 1.0)
        else:
            # Degree-like integer values
            n_bins = int(np.ceil(vmax)) + 1
            n_bins = max(n_bins, 2)
            bin_range = (0.0, float(n_bins))

        def to_hist(values):
            values = np.asarray(values, dtype=np.float64)
            if len(values) == 0:
                return np.zeros(n_bins, dtype=np.float64)
            hist, _ = np.histogram(values, bins=n_bins, range=bin_range, density=False)
            s = hist.sum()
            if s > 0:
                hist = hist.astype(np.float64) / s
            return hist.astype(np.float64)

        vecs1 = [to_hist(s) for s in samples1]
        vecs2 = [to_hist(s) for s in samples2]
    else:
        vecs1 = [np.asarray(s, dtype=np.float64) for s in samples1]
        vecs2 = [np.asarray(s, dtype=np.float64) for s in samples2]

    n = len(vecs1)
    m = len(vecs2)
    if n == 0 or m == 0:
        return 0.0

    xx = 0.0
    for i in range(n):
        for j in range(n):
            xx += kfn(vecs1[i], vecs1[j])
    xx /= n * n

    yy = 0.0
    for i in range(m):
        for j in range(m):
            yy += kfn(vecs2[i], vecs2[j])
    yy /= m * m

    xy = 0.0
    for i in range(n):
        for j in range(m):
            xy += kfn(vecs1[i], vecs2[j])
    xy /= n * m

    return xx + yy - 2 * xy


def evaluate_graphs(gen_graphs, ref_graphs, n_eval=None):
    """Evaluate generated graphs against reference graphs using MMD.

    Returns dict with mmd_degree, mmd_clustering, mmd_orbit.
    """
    if n_eval is not None:
        gen_graphs = gen_graphs[:n_eval]

    # Filter out empty graphs
    gen_graphs = [G for G in gen_graphs if G.number_of_nodes() > 0 and G.number_of_edges() > 0]
    ref_graphs = [G for G in ref_graphs if G.number_of_nodes() > 0 and G.number_of_edges() > 0]

    if len(gen_graphs) == 0:
        return {"mmd_degree": 10.0, "mmd_clustering": 10.0, "mmd_orbit": 10.0}

    # Compute statistics for reference graphs
    ref_degree = [degree_stats(G) for G in ref_graphs]
    ref_cluster = [clustering_stats(G) for G in ref_graphs]
    ref_orbit = [orbit_stats(G) for G in ref_graphs]

    # Compute statistics for generated graphs
    gen_degree = [degree_stats(G) for G in gen_graphs]
    gen_cluster = [clustering_stats(G) for G in gen_graphs]
    gen_orbit = [orbit_stats(G) for G in gen_graphs]

    # Compute MMD for each statistic (GDSS conventions)
    # degree: histogrammed integer degrees, gaussian_emd kernel, sigma=1.0
    mmd_deg = compute_mmd(ref_degree, gen_degree, kernel="gaussian_emd", sigma=1.0, is_hist=True)
    # clustering: histogrammed over [0,1], gaussian_emd kernel, sigma=1.0 / bins=100
    mmd_clus = compute_mmd(ref_cluster, gen_cluster, kernel="gaussian_emd", sigma=1.0 / 10.0, is_hist=True)
    # orbit: raw per-graph orbit count vectors, gaussian kernel, sigma=30.0
    mmd_orb = compute_mmd(ref_orbit, gen_orbit, kernel="gaussian", sigma=30.0, is_hist=False)

    return {
        "mmd_degree": float(mmd_deg),
        "mmd_clustering": float(mmd_clus),
        "mmd_orbit": float(mmd_orb),
    }


# ============================================================================
# EDITABLE REGION START (lines 446-590)
# ============================================================================
# The agent should modify the GraphGenerator class below.
# The class must implement:
#   - __init__(self, max_nodes, **kwargs): initialize model parameters
#   - train_step(self, adj, node_counts) -> dict: one training step, returns loss dict
#   - sample(self, n_samples, device) -> (adj_matrices, node_counts):
#       generate n_samples graphs, return adjacency tensors and node count tensors
#
# The model receives adjacency matrices [B, max_nodes, max_nodes] and node counts [B].
# It should generate adjacency matrices of similar structure.
# ============================================================================

class GraphGenerator(nn.Module):
    """Generative model for graphs.

    This is a simple VAE-based baseline that encodes graphs into a latent space
    and decodes to adjacency matrices. The agent should replace this with a
    better generative model.

    Design space includes:
      - Autoregressive models (node-by-node or edge-by-edge generation)
      - One-shot models (generate full adjacency at once)
      - Score-based / diffusion models (iterative denoising)
      - Flow-based models (invertible transformations)
      - VAE variants with graph-aware encoders/decoders

    Args:
        max_nodes: Maximum number of nodes in generated graphs.
        hidden_dim: Hidden dimension for encoder/decoder networks.
        latent_dim: Latent space dimension.
        lr: Learning rate for the optimizer.
    """

    def __init__(self, max_nodes, hidden_dim=256, latent_dim=64, lr=1e-3, **kwargs):
        super().__init__()
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        adj_size = max_nodes * max_nodes

        # Encoder: adjacency -> latent
        self.encoder = nn.Sequential(
            nn.Linear(adj_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: latent -> adjacency
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, adj_size),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def encode(self, adj):
        """Encode adjacency matrix to latent distribution parameters."""
        B = adj.shape[0]
        x = adj.view(B, -1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent vector to adjacency matrix."""
        B = z.shape[0]
        logits = self.decoder(z)
        logits = logits.view(B, self.max_nodes, self.max_nodes)
        # Make symmetric
        logits = (logits + logits.transpose(1, 2)) / 2
        # Zero diagonal (no self-loops)
        mask = 1 - torch.eye(self.max_nodes, device=z.device).unsqueeze(0)
        logits = logits * mask
        return logits

    def train_step(self, adj, node_counts):
        """One training step.

        Args:
            adj: [B, max_nodes, max_nodes] binary adjacency matrices.
            node_counts: [B] number of actual nodes per graph.

        Returns:
            dict with at least 'loss' key (float).
        """
        self.train()
        self.optimizer.zero_grad()

        mu, logvar = self.encode(adj)
        z = self.reparameterize(mu, logvar)
        adj_logits = self.decode(z)

        # Reconstruction loss (binary cross-entropy)
        recon_loss = F.binary_cross_entropy_with_logits(
            adj_logits, adj, reduction="mean"
        )

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recon_loss + 0.001 * kl_loss
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
        }

    def sample(self, n_samples, device):
        """Generate graphs by sampling from the latent space.

        Args:
            n_samples: Number of graphs to generate.
            device: torch device.

        Returns:
            adj: Tensor [n_samples, max_nodes, max_nodes] — generated adjacency
                 matrices (binary, symmetric, no self-loops).
            node_counts: Tensor [n_samples] — number of nodes per generated graph.
                         Can use max_nodes if variable-size generation is not supported.
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=device)
            adj_logits = self.decode(z)
            adj = (torch.sigmoid(adj_logits) > 0.5).float()
            # Estimate node counts from generated adjacency
            # (nodes with at least one edge are considered present)
            node_mask = (adj.sum(dim=-1) > 0).float()  # [B, max_nodes]
            node_counts = node_mask.sum(dim=-1).long()  # [B]
            node_counts = torch.clamp(node_counts, min=2)
        return adj, node_counts

# ============================================================================
# EDITABLE REGION END (lines 446-590)
# ============================================================================


# ============================================================================
# Training & Evaluation Loop (FIXED)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Graph Generation Benchmark")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["community_small", "ego_small", "enzymes"])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--n-gen", type=int, default=None,
                        help="Number of graphs to generate for evaluation (default: same as dataset)")
    args = parser.parse_args()

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load dataset
    print(f"Loading dataset: {args.dataset}", flush=True)
    graphs = load_dataset(args.dataset)
    n_graphs = len(graphs)
    max_nodes = max(G.number_of_nodes() for G in graphs)
    print(f"Loaded {n_graphs} graphs, max_nodes={max_nodes}", flush=True)

    # Split: 80% train, 20% test (reference for evaluation)
    random.shuffle(graphs)
    n_train = int(0.8 * n_graphs)
    train_graphs = graphs[:n_train]
    test_graphs = graphs[n_train:]
    print(f"Train: {len(train_graphs)}, Test (reference): {len(test_graphs)}", flush=True)

    # Convert to adjacency matrices
    train_adjs, train_counts = graphs_to_adj(train_graphs, max_nodes)
    train_adjs = train_adjs.to(device)
    train_counts = train_counts.to(device)

    train_dataset = GraphDataset(train_adjs, train_counts)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Build model
    model = GraphGenerator(max_nodes=max_nodes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}", flush=True)

    # ── Parameter Budget Check ──
    # Budget = 1.05x largest baseline. Baselines include:
    # - DiGress (graph transformer denoiser, scales with max_nodes linearly)
    # - GraphVAE (VAE with adjacency decoder, scales with max_nodes^2)
    # - MoFlow (normalizing flow on adjacency, scales with max_nodes^2 via tri_size)
    # On large-node datasets (enzymes, max_nodes=125), adjacency-based methods
    # dominate: MoFlow ~20M params vs DiGress ~900K.
    _H_dg = 128
    _n_gt_layers = 4
    _gt_per_layer = 4 * (_H_dg * _H_dg + _H_dg) + 8 + 4 * _H_dg + 8 * _H_dg * _H_dg + 5 * _H_dg
    _digress_params = (
        max_nodes * _H_dg + _H_dg                          # node_embed
        + _H_dg * _H_dg + 3 * _H_dg + 1                   # time_embed
        + _n_gt_layers * _gt_per_layer                      # transformer layers
        + (2 * _H_dg + 1) * _H_dg + _H_dg + _H_dg + 1     # edge_pred
        + _H_dg * (_H_dg // 2) + (_H_dg // 2) + (_H_dg // 2) + 1  # node_pred
        + 5000                                              # optimizer states, misc
    )
    # GraphVAE: GCN encoder + VAE + MLP decoder to adj_size = max_nodes^2
    _gvae_h = 256
    _gvae_lat = 64
    _adj_size = max_nodes * max_nodes
    _graphvae_params = (
        max_nodes * _gvae_h + _gvae_h                      # GCN1(max_nodes -> H)
        + _gvae_h * _gvae_h + _gvae_h                      # GCN2(H -> H)
        + 2 * (_gvae_h * _gvae_lat + _gvae_lat)            # fc_mu + fc_logvar
        + _gvae_lat * _gvae_h + _gvae_h                    # decoder L1
        + _gvae_h * _gvae_h + _gvae_h                      # decoder L2
        + _gvae_h * _adj_size + _adj_size                   # decoder L3 (H -> N^2)
        + _gvae_lat * (_gvae_h // 2) + (_gvae_h // 2)      # node_pred L1
        + (_gvae_h // 2) * max_nodes + max_nodes            # node_pred L2
    )
    # MoFlow: normalizing flow on upper-triangular adjacency (tri_size = N*(N-1)/2)
    _tri_size = max_nodes * (max_nodes - 1) // 2
    _mf_h = 256
    _mf_half = _tri_size // 2
    _mf_other = _tri_size - _mf_half
    _mf_n_layers = 6
    # Per AffineCoupling: Linear(half, H) + Linear(H, H) + Linear(H, 2*other)
    _mf_coupling = (
        _mf_half * _mf_h + _mf_h
        + _mf_h * _mf_h + _mf_h
        + _mf_h * 2 * _mf_other + 2 * _mf_other
    )
    # Per ActNorm: 2 * tri_size
    _mf_per_block = _mf_coupling + 2 * _tri_size
    _moflow_params = (
        _mf_n_layers * _mf_per_block                        # flow blocks
        + _tri_size * _mf_h + _mf_h                         # node_pred L1
        + _mf_h * max_nodes + max_nodes                     # node_pred L2
    )
    _max_baseline = max(_digress_params, _graphvae_params, _moflow_params)
    _param_budget = int(_max_baseline * 1.05)
    print(f"Parameter budget: {n_params:,} / {_param_budget:,} (1.05x largest baseline)", flush=True)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        epoch_losses = defaultdict(float)
        n_batches = 0
        for batch_adj, batch_counts in train_loader:
            batch_adj = batch_adj.to(device)
            batch_counts = batch_counts.to(device)
            loss_dict = model.train_step(batch_adj, batch_counts)
            for k, v in loss_dict.items():
                epoch_losses[k] += v
            n_batches += 1

        if epoch % 500 == 0 or epoch == 1:
            avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
            loss_str = " ".join(f"{k}={v:.6f}" for k, v in avg_losses.items())
            print(f"TRAIN_METRICS epoch={epoch} {loss_str}", flush=True)

    # Generate and evaluate
    n_gen = args.n_gen if args.n_gen is not None else len(test_graphs)
    n_gen = max(n_gen, len(test_graphs))  # At least as many as test set
    print(f"Generating {n_gen} graphs for evaluation...", flush=True)

    gen_adjs, gen_counts = model.sample(n_gen, device)
    gen_graphs = adj_to_graphs(gen_adjs, gen_counts)

    # Compute metrics
    metrics = evaluate_graphs(gen_graphs, test_graphs)

    # Print final metrics
    mmd_avg = np.mean([metrics["mmd_degree"], metrics["mmd_clustering"], metrics["mmd_orbit"]])
    metrics["mmd_avg"] = float(mmd_avg)

    metrics_str = " ".join(f"{k}={v:.6f}" for k, v in metrics.items())
    print(f"TEST_METRICS {metrics_str}", flush=True)

    # Save generated graphs
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(
        {"gen_adjs": gen_adjs.cpu(), "gen_counts": gen_counts.cpu()},
        os.path.join(args.output_dir, "generated_graphs.pt"),
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
