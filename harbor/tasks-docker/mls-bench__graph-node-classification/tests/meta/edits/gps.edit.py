"""GPS (General, Powerful, Scalable Graph Transformer) baseline for graph-node-classification.

Reference: Rampasek et al., "Recipe for a General, Powerful, Scalable Graph Transformer", NeurIPS 2022.
Combines local MPNN with global self-attention in each layer.
"""

_FILE = "pytorch-geometric/custom_nodecls.py"

_CONTENT = """\
class CustomMessagePassingLayer(MessagePassing):
    \"\"\"GPS baseline: combined local MPNN + global multi-head self-attention.

    Each GPS layer applies:
    1. Local message passing (GCN-style) on the graph
    2. Global multi-head self-attention over all nodes
    3. Residual connection + layer norm + FFN
    \"\"\"

    def __init__(self, in_channels: int, out_channels: int,
                 heads: int = 4, attn_dropout: float = 0.2):
        super().__init__(aggr="add")
        self.heads = heads
        self.out_channels = out_channels

        # Input projection (if dimensions differ)
        self.lin_in = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

        # Local MPNN component (GCN-style)
        self.lin_msg = nn.Linear(out_channels, out_channels, bias=False)
        self.lin_update = nn.Linear(out_channels, out_channels)

        # Global attention component; higher attn_dropout reduces seed-to-seed
        # variance observed on CiteSeer (seed=42 collapsed under 0.1 dropout).
        self.attn = nn.MultiheadAttention(out_channels, heads,
                                          dropout=attn_dropout, batch_first=True)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(out_channels * 2, out_channels),
            nn.Dropout(0.2),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.norm3 = nn.LayerNorm(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.lin_in, nn.Linear):
            nn.init.xavier_uniform_(self.lin_in.weight)
            nn.init.zeros_(self.lin_in.bias)
        nn.init.xavier_uniform_(self.lin_msg.weight)
        nn.init.xavier_uniform_(self.lin_update.weight)
        nn.init.zeros_(self.lin_update.bias)
        # Explicit init of attention and FFN projections for reproducibility
        # (reduces seed sensitivity on small graphs)
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.zeros_(self.attn.in_proj_bias)
        nn.init.xavier_uniform_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        for m in self.ffn:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        x = self.lin_in(x)

        # 1. Local message passing
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index_sl
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        local_out = self.propagate(edge_index_sl, x=self.lin_msg(x), norm=norm)
        local_out = self.lin_update(local_out)

        # Residual + norm after local
        x = self.norm1(x + local_out)

        # 2. Global self-attention (treat all nodes as a single sequence)
        x_unsq = x.unsqueeze(0)  # [1, N, D]
        attn_out, _ = self.attn(x_unsq, x_unsq, x_unsq)
        attn_out = attn_out.squeeze(0)  # [N, D]

        # Residual + norm after global attention
        x = self.norm2(x + attn_out)

        # 3. FFN with residual
        x = self.norm3(x + self.ffn(x))

        return x

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j


class CustomGNN(nn.Module):
    \"\"\"GPS model: stacked GPS layers with local+global processing.

    Uses a lower learning rate (0.001) and moderate weight decay to
    stabilise transformer-style self-attention training on small
    citation networks.
    \"\"\"

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        # Lower LR + higher WD to stabilise attention on small Planetoid graphs.
        # Previously 1e-3 caused seed=42 collapse on CiteSeer (0.319 vs ~0.6).
        self.custom_lr = 5e-4
        self.custom_wd = 1e-3
        self.convs = nn.ModuleList()
        self.convs.append(CustomMessagePassingLayer(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(CustomMessagePassingLayer(hidden_channels, hidden_channels))
        self.classifier = nn.Linear(hidden_channels, out_channels)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
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
