"""NAGphormer baseline for graph-node-classification.

Reference: Chen et al., "NAGphormer: A Tokenized Graph Transformer for Node Classification
in Large Graphs", ICLR 2023.
Hop2Token: constructs per-node token sequences from multi-hop neighborhood aggregation,
then applies a Transformer encoder with attention-based readout.
"""

_FILE = "pytorch-geometric/custom_nodecls.py"

_CONTENT = """\
class CustomMessagePassingLayer(MessagePassing):
    \"\"\"NAGphormer-style Hop2Token module.

    Aggregates neighborhood features at multiple hops using GCN-style
    symmetric normalization (D^{-1/2} A D^{-1/2}) to produce per-node
    token sequences. Serves as the tokenization front-end.

    Reference: Chen et al., "NAGphormer: A Tokenized Graph Transformer
    for Node Classification in Large Graphs", ICLR 2023.
    \"\"\"

    def __init__(self, in_channels: int, out_channels: int, num_hops: int = 5):
        super().__init__(aggr="add")
        self.num_hops = num_hops
        self.out_channels = out_channels
        # Single shared projection from input features to hidden dim
        self.lin = nn.Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        \"\"\"Return stacked hop tokens [N, num_hops+1, D].\"\"\"
        # Shared projection
        x_proj = self.lin(x)

        # Precompute GCN-style normalization with self-loops
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index_sl
        deg = degree(col, x_proj.size(0), dtype=x_proj.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        self._norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        self._edge_index_sl = edge_index_sl

        hop_features = [x_proj]  # hop 0 = self features

        h = x_proj
        for k in range(1, self.num_hops + 1):
            # GCN-style normalized propagation
            h = self.propagate(self._edge_index_sl, x=h, norm=self._norm)
            hop_features.append(h)

        return torch.stack(hop_features, dim=1)  # [N, num_hops+1, D]

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j


class CustomGNN(nn.Module):
    \"\"\"NAGphormer model: Hop2Token + Transformer encoder + hop-attention readout.

    Architecture follows the original paper:
    1. Multi-hop aggregation produces per-node token sequences
    2. Learnable hop-type embeddings are added
    3. Transformer encoder processes each node's token sequence
    4. Weighted attention readout aggregates tokens to node embedding
    5. Classification head produces logits
    \"\"\"

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        # Paper (Chen et al., ICLR 2023) uses dropout ~0.1-0.3 on Planetoid
        self.dropout = 0.1
        # Paper sweeps K in {3,...,10}; 7 hops works well for Cora/CiteSeer/PubMed
        self.num_hops = 7
        self.hidden_channels = hidden_channels
        # Transformer-based model needs lower lr than GCN; paper uses ~5e-4
        # with ~1e-5 weight decay on Planetoid
        self.custom_lr = 5e-4
        self.custom_wd = 1e-5

        # Hop2Token tokenization
        self.tokenizer = CustomMessagePassingLayer(
            in_channels, hidden_channels, num_hops=self.num_hops)

        # Learnable hop-type embedding
        self.hop_embedding = nn.Parameter(
            torch.zeros(1, self.num_hops + 1, hidden_channels))
        nn.init.normal_(self.hop_embedding, std=0.02)

        # Input layer norm
        self.input_norm = nn.LayerNorm(hidden_channels)

        # Transformer encoder layers; paper uses 8 heads and FFN=2*d
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels, nhead=8,
            dim_feedforward=hidden_channels * 2,
            dropout=self.dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(hidden_channels),
        )

        # Attention readout over hop tokens
        self.attn_vec = nn.Linear(hidden_channels, 1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        N = x.size(0)

        # Tokenize: [N, num_hops+1, D]
        tokens = self.tokenizer(x, edge_index)
        tokens = tokens + self.hop_embedding

        # Input normalization
        tokens = self.input_norm(tokens)
        tokens = F.dropout(tokens, p=self.dropout, training=self.training)

        # Transformer encoding (each node's token sequence independently)
        tokens = self.transformer(tokens)

        # Attention-weighted readout over hop tokens
        attn_scores = self.attn_vec(tokens).squeeze(-1)  # [N, num_hops+1]
        attn_weights = F.softmax(attn_scores, dim=-1)    # [N, num_hops+1]
        node_repr = (attn_weights.unsqueeze(-1) * tokens).sum(dim=1)  # [N, D]

        return self.classifier(node_repr)
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
