"""GAT (Graph Attention Network) baseline for graph-node-classification.

Reference: Velickovic et al., "Graph Attention Networks", ICLR 2018.
Multi-head attention mechanism for message weighting.
"""

_FILE = "pytorch-geometric/custom_nodecls.py"

_CONTENT = """\
class CustomMessagePassingLayer(MessagePassing):
    \"\"\"GAT baseline: graph attention layer with multi-head attention.\"\"\"

    def __init__(self, in_channels: int, out_channels: int,
                 heads: int = 8, concat: bool = True,
                 negative_slope: float = 0.2):
        super().__init__(aggr="add", node_dim=0)
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope

        if concat:
            assert out_channels % heads == 0
            self.head_dim = out_channels // heads
        else:
            self.head_dim = out_channels

        self.lin = nn.Linear(in_channels, heads * self.head_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(heads * self.head_dim if concat else out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        H, D = self.heads, self.head_dim
        x = self.lin(x).view(-1, H, D)

        # Compute attention coefficients
        alpha_src = (x * self.att_src).sum(dim=-1)  # [N, H]
        alpha_dst = (x * self.att_dst).sum(dim=-1)  # [N, H]

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.propagate(edge_index, x=x,
                             alpha_src=alpha_src, alpha_dst=alpha_dst)

        if self.concat:
            out = out.view(-1, H * D)
        else:
            out = out.mean(dim=1)

        out = out + self.bias
        return out

    def message(self, x_j: Tensor, alpha_src_i: Tensor,
                alpha_dst_j: Tensor, index: Tensor,
                ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        alpha = alpha_src_i + alpha_dst_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=0.6, training=self.training)
        return x_j * alpha.unsqueeze(-1)


class CustomGNN(nn.Module):
    \"\"\"GAT model: multi-head attention GNN.\"\"\"

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2,
                 dropout: float = 0.6):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        # First layer: 8 heads, concat
        self.convs.append(CustomMessagePassingLayer(
            in_channels, hidden_channels, heads=8, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(CustomMessagePassingLayer(
                hidden_channels, hidden_channels, heads=8, concat=True))
        # Last layer: 1 head, no concat (average)
        self.convs.append(CustomMessagePassingLayer(
            hidden_channels, out_channels, heads=1, concat=False))

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.elu(x)
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
