"""GraphSAGE baseline for graph-node-classification.

Reference: Hamilton et al., "Inductive Representation Learning on Large Graphs", NeurIPS 2017.
Mean aggregation with neighborhood sampling.
"""

_FILE = "pytorch-geometric/custom_nodecls.py"

_CONTENT = """\
class CustomMessagePassingLayer(MessagePassing):
    \"\"\"GraphSAGE baseline: mean-aggregation message passing.\"\"\"

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="mean")
        self.lin_self = nn.Linear(in_channels, out_channels, bias=True)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.xavier_uniform_(self.lin_neigh.weight)
        nn.init.zeros_(self.lin_self.bias)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        # Aggregate neighbor features (mean)
        neigh_agg = self.propagate(edge_index, x=x)
        # Combine self and neighbor
        out = self.lin_self(x) + self.lin_neigh(neigh_agg)
        out = F.normalize(out, p=2, dim=-1)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j


class CustomGNN(nn.Module):
    \"\"\"GraphSAGE model: mean-aggregation GNN with L2 normalization.\"\"\"

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(CustomMessagePassingLayer(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(CustomMessagePassingLayer(hidden_channels, hidden_channels))
        self.convs.append(CustomMessagePassingLayer(hidden_channels, out_channels))

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 48,
        "end_line": 157,
        "content": _CONTENT,
    },
]
