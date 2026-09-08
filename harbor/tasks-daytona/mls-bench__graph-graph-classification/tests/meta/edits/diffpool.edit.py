"""DiffPool (Differentiable Pooling) readout baseline.

Learns a differentiable soft cluster assignment matrix to hierarchically
coarsen graphs. From "Hierarchical Graph Representation Learning with
Differentiable Pooling" (Ying et al., NeurIPS 2018).

Reference: torch_geometric.nn.dense_diff_pool
Reported: MUTAG ~82-90, PROTEINS ~76, NCI1 ~74-82
(SOTA hierarchical pooling; often improves with proper tuning)
"""

_FILE = "pytorch-geometric/custom_graph_cls.py"

_CONTENT = """\
class GraphReadout(nn.Module):
    \"\"\"DiffPool Readout (Ying et al., 2018).

    Uses a learned soft assignment matrix to cluster nodes into
    a fixed number of super-nodes, then reads out from the
    coarsened graph. Two-level hierarchy.
    \"\"\"

    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Assignment network: maps nodes to clusters
        self.max_nodes = 150  # Max nodes per graph (padded)
        self.num_clusters = 25
        self.assign_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_clusters),
        )
        self.output_dim = hidden_dim

    def forward(self, x, edge_index, batch, layer_outputs):
        # Convert to dense batch format
        x_dense, mask = to_dense_batch(x, batch)  # [B, N_max, D]
        adj = to_dense_adj(edge_index, batch)  # [B, N_max, N_max]

        # Compute soft assignment
        s = self.assign_nn(x_dense)  # [B, N_max, K]
        s = s.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        s = torch.softmax(s, dim=1)
        s = s * mask.unsqueeze(-1).float()

        # Pool: X_coarse = S^T @ X, A_coarse = S^T @ A @ S
        x_coarse = torch.bmm(s.transpose(1, 2), x_dense)  # [B, K, D]

        # Global mean pool over clusters
        return x_coarse.mean(dim=1)  # [B, D]
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 41,
        "end_line": 81,
        "content": _CONTENT,
    },
]
