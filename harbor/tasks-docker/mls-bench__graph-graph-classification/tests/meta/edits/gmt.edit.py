"""GMT (Graph Multiset Transformer) readout baseline.

Multi-head attention based global pooling that captures node interactions
and structural dependencies. From "Accurate Learning of Graph Representations
with Graph Multiset Pooling" (Baek et al., ICLR 2021).

Reference: torch_geometric.nn.aggr.GraphMultisetTransformer
Reported: MUTAG ~83-89, PROTEINS ~75-78, NCI1 ~78-82
(SOTA attention-based readout; captures structural dependencies)
"""

_FILE = "pytorch-geometric/custom_graph_cls.py"

_CONTENT = """\
class GraphReadout(nn.Module):
    \"\"\"GMT Readout (Baek et al., 2021).

    Graph Multiset Transformer: uses multi-head attention to
    aggregate node features with structure awareness.
    Combines PMA (Pooling by Multihead Attention) with
    graph structure information.
    \"\"\"

    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = hidden_dim
        self.num_heads = 4
        self.num_seeds = 1  # Number of seed vectors for PMA

        # Seed vector for pooling-by-attention
        self.seed = nn.Parameter(torch.randn(1, self.num_seeds, hidden_dim))
        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            hidden_dim, self.num_heads, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, batch, layer_outputs):
        # Convert to dense batch
        x_dense, mask = to_dense_batch(x, batch)  # [B, N_max, D]
        B = x_dense.size(0)

        # Expand seed vectors for batch
        seeds = self.seed.expand(B, -1, -1)  # [B, 1, D]

        # Cross-attention: seeds attend to nodes
        # key_padding_mask: True means ignore
        key_pad = ~mask  # [B, N_max]
        out, _ = self.attn(seeds, x_dense, x_dense,
                           key_padding_mask=key_pad)  # [B, 1, D]
        out = self.norm1(out + seeds)
        out = self.norm2(out + self.ffn(out))

        return out.squeeze(1)  # [B, D]
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
