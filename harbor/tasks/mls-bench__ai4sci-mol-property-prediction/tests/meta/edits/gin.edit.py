"""GIN baseline — Graph Isomorphism Network.
Uses the template starter GIN implementation.
Reference: Xu et al., "How Powerful are Graph Neural Networks?" (ICLR 2019)
"""

_FILE = "Uni-Mol/custom_molprop.py"

_CONTENT = """\

class GINConv(nn.Module):
    \"\"\"Graph Isomorphism Network convolution layer.\"\"\"

    def __init__(self, in_dim, out_dim, edge_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.edge_proj = nn.Linear(edge_dim, in_dim)
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, edge_attr, batch_idx):
        \"\"\"
        x: [total_atoms, in_dim]
        edge_index: [2, total_edges]
        edge_attr: [total_edges, edge_dim]
        batch_idx: [total_atoms]
        \"\"\"
        src, dst = edge_index
        edge_msg = self.edge_proj(edge_attr)
        msg = x[src] + edge_msg

        # Aggregate messages to destination nodes
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)

        out = self.mlp((1 + self.eps) * x + agg)
        return out


class MoleculeModel(nn.Module):
    \"\"\"Starter model: Graph Isomorphism Network (GIN) with mean pooling.

    Simple but effective baseline for molecular property prediction.
    Uses message passing on the molecular graph with learned edge features.
    \"\"\"

    def __init__(self, atom_dim: int, edge_dim: int, num_tasks: int, task_type: str):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_type = task_type
        hidden_dim = 256
        num_layers = 4

        self.atom_embed = nn.Linear(atom_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GINConv(hidden_dim, hidden_dim, edge_dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.1)

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_tasks),
        )

    def forward(self, batch):
        \"\"\"
        Args:
            batch: MolBatch with sparse graph data.
        Returns:
            predictions: [B, num_tasks]
        \"\"\"
        x = self.atom_embed(batch.x)

        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, batch.edge_index, batch.edge_attr, batch.batch_idx)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x = x + self.dropout(x_new)  # residual

        # Mean pooling per graph
        num_graphs = batch.batch_idx.max().item() + 1
        graph_embed = torch.zeros(num_graphs, x.size(-1), device=x.device)
        counts = torch.zeros(num_graphs, 1, device=x.device)
        graph_embed.index_add_(0, batch.batch_idx, x)
        counts.index_add_(0, batch.batch_idx, torch.ones(x.size(0), 1, device=x.device))
        graph_embed = graph_embed / counts.clamp(min=1)

        return self.readout(graph_embed)

"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 115,
        "end_line": 207,
        "content": _CONTENT,
    },
]
