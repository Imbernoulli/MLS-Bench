"""GIN Sum Readout baseline.

Standard GIN readout using sum pooling over all layer outputs (Jumping
Knowledge concatenation + sum). This is the original GIN readout from
"How Powerful are Graph Neural Networks?" (Xu et al., ICLR 2019).

Reference: global_add_pool applied to concatenated multi-layer outputs.
Reported: MUTAG ~89.4, PROTEINS ~76.2, NCI1 ~82.7
"""

_FILE = "pytorch-geometric/custom_graph_cls.py"

_CONTENT = """\
class GraphReadout(nn.Module):
    \"\"\"GIN JK-Sum Readout (Xu et al., 2019).

    Concatenates sum-pooled embeddings from all GIN layers
    (Jumping Knowledge). Each layer's graph embedding is batch-normalized
    before concatenation to stabilize training -- this prevents the
    different-scale representations across layers from causing
    optimization issues (some folds failing to converge).

    The output dimension is hidden_dim * num_layers, matching the
    original GIN paper's readout.
    \"\"\"

    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Full concatenated dimension -- no projection bottleneck
        self.output_dim = hidden_dim * num_layers
        # Per-layer batch normalization on graph-level embeddings
        self.graph_bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, x, edge_index, batch, layer_outputs):
        # Sum-pool each layer's node embeddings independently
        graph_embs = []
        for i, h in enumerate(layer_outputs):
            g = global_add_pool(h, batch)
            g = self.graph_bns[i](g)
            graph_embs.append(g)
        # Concatenate all layers (Jumping Knowledge)
        return torch.cat(graph_embs, dim=-1)
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
