"""Set2Set readout baseline.

LSTM-based attention readout that processes the node set in multiple steps,
learning which nodes to attend to. From "Order Matters: Sequence to sequence
for sets" (Vinyals et al., ICLR 2016).

Reference: torch_geometric.nn.aggr.Set2Set
Reported: Competitive on molecular datasets; commonly used with MPNN.
"""

_FILE = "pytorch-geometric/custom_graph_cls.py"

_CONTENT = """\
class GraphReadout(nn.Module):
    \"\"\"Set2Set Readout (Vinyals et al., 2016).

    LSTM-based attention mechanism that iteratively attends to node
    embeddings, producing a 2*hidden_dim output per graph.
    \"\"\"

    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.processing_steps = 6
        from torch_geometric.nn.aggr import Set2Set as Set2SetAggr
        self.set2set = Set2SetAggr(hidden_dim, self.processing_steps)
        # Set2Set outputs 2 * hidden_dim; project back
        self.proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.output_dim = hidden_dim

    def forward(self, x, edge_index, batch, layer_outputs):
        out = self.set2set(x, batch)  # [B, 2 * hidden_dim]
        return self.proj(out)
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
