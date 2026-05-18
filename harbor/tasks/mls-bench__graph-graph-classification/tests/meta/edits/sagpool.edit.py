"""SAGPool (Self-Attention Graph Pooling) readout baseline.

Hierarchical graph pooling using self-attention scores to select top-k
informative nodes, then applies readout on the coarsened graph. From
"Self-Attention Graph Pooling" (Lee et al., ICML 2019).

Reference: torch_geometric.nn.pool.SAGPooling
Reported: MUTAG ~73, PROTEINS ~74, NCI1 ~74-80
(SOTA hierarchical pooling at time of publication)
"""

_FILE = "pytorch-geometric/custom_graph_cls.py"

_CONTENT = """\
class GraphReadout(nn.Module):
    \"\"\"SAGPool Hierarchical Readout (Lee et al., 2019).

    Uses self-attention scores to hierarchically select top-k nodes,
    then applies sum+mean global readout on the coarsened graph.
    Two-level hierarchy: original -> coarsened.
    \"\"\"

    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        from torch_geometric.nn.pool import SAGPooling
        self.pool1 = SAGPooling(hidden_dim, ratio=0.5)
        self.pool2 = SAGPooling(hidden_dim, ratio=0.5)
        # 3 levels (original + 2 coarsened), each with sum+mean
        self.output_dim = hidden_dim * 2 * 3
        self.proj = nn.Linear(self.output_dim, hidden_dim)
        self.output_dim = hidden_dim

    def forward(self, x, edge_index, batch, layer_outputs):
        # Level 0: readout on original graph
        r0 = torch.cat([global_add_pool(x, batch),
                         global_mean_pool(x, batch)], dim=-1)

        # Level 1: first coarsening
        x1, edge_index1, _, batch1, perm1, score1 = self.pool1(
            x, edge_index, batch=batch)
        r1 = torch.cat([global_add_pool(x1, batch1),
                         global_mean_pool(x1, batch1)], dim=-1)

        # Level 2: second coarsening
        x2, edge_index2, _, batch2, perm2, score2 = self.pool2(
            x1, edge_index1, batch=batch1)
        r2 = torch.cat([global_add_pool(x2, batch2),
                         global_mean_pool(x2, batch2)], dim=-1)

        return self.proj(torch.cat([r0, r1, r2], dim=-1))
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
