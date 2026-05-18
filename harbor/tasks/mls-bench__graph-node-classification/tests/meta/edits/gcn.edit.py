"""GCN (Graph Convolutional Network) baseline for graph-node-classification.

Reference: Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017.
Uses PyG's built-in GCNConv layer.
"""

_FILE = "pytorch-geometric/custom_nodecls.py"

_CONTENT = """\
class CustomMessagePassingLayer(MessagePassing):
    \"\"\"GCN baseline: standard graph convolutional layer.\"\"\"

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        x = self.lin(x)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = self.propagate(edge_index, x=x, norm=norm)
        out = out + self.bias
        return out

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j


class CustomGNN(nn.Module):
    \"\"\"GCN model: 2-layer GCN with ReLU and dropout.\"\"\"

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
