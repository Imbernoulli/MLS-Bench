"""RevGAT baseline adapted for graph-node-classification.

Reference: Li et al., "Training Graph Neural Networks with 1000 Layers", ICML 2021.
Uses reversible connections with GAT layers for moderate-depth citation
network benchmarks in this harness.
"""

_FILE = "pytorch-geometric/custom_nodecls.py"

_CONTENT = """\
class CustomMessagePassingLayer(MessagePassing):
    \"\"\"GAT layer with group normalization for reversible-style stacking.\"\"\"

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

        # Group normalization for stable deep training
        out_dim = heads * self.head_dim if concat else out_channels
        num_groups = min(8, out_dim)
        while out_dim % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        H, D = self.heads, self.head_dim
        x_proj = self.lin(x).view(-1, H, D)

        alpha_src = (x_proj * self.att_src).sum(dim=-1)
        alpha_dst = (x_proj * self.att_dst).sum(dim=-1)

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.propagate(edge_index, x=x_proj,
                             alpha_src=alpha_src, alpha_dst=alpha_dst)

        if self.concat:
            out = out.view(-1, H * D)
        else:
            out = out.mean(dim=1)

        out = out + self.bias
        out = self.norm(out)
        return out

    def message(self, x_j: Tensor, alpha_src_i: Tensor,
                alpha_dst_j: Tensor, index: Tensor,
                ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        alpha = alpha_src_i + alpha_dst_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=0.6, training=self.training)
        return x_j * alpha.unsqueeze(-1)


class _RevBlock(nn.Module):
    \"\"\"Reversible block: F and G are two sub-functions.
    x1_out = x1 + F(x2), x2_out = x2 + G(x1_out)
    \"\"\"

    def __init__(self, channels, heads=8, negative_slope=0.2, dropout=0.5):
        super().__init__()
        self.F_conv = CustomMessagePassingLayer(
            channels, channels, heads=heads, concat=True, negative_slope=negative_slope)
        self.G_ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x1, x2, edge_index):
        y1 = x1 + self.F_conv(x2, edge_index)
        y2 = x2 + self.G_ffn(y1)
        return y1, y2


class CustomGNN(nn.Module):
    \"\"\"RevGAT-style reversible GAT model with group norm.\"\"\"

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        half_dim = hidden_channels // 2

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # Reversible blocks (deeper architecture)
        depth = max(num_layers, 4)
        self.rev_blocks = nn.ModuleList([
            _RevBlock(half_dim, heads=4, dropout=dropout)
            for _ in range(depth)
        ])

        # Classifier
        self.norm = nn.LayerNorm(hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        x = self.input_proj(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Split into two streams for reversible blocks
        x1, x2 = x.chunk(2, dim=-1)

        for block in self.rev_blocks:
            x1, x2 = block(x1, x2, edge_index)

        # Merge streams
        x = torch.cat([x1, x2], dim=-1)
        x = self.norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)
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
