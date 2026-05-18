"""GCN + MLP decoder baseline for graph-link-prediction.

Multi-layer GCN encoder with batch normalization + MLP decoder (pair features).
This matches the standard OGB GCN+MLP link prediction baseline which achieves
~55 Hits@50 on ogbl-collab. Pure dot-product decoder is strictly weaker.

Reference: OGB leaderboard, SEAL/BUDDY paper Table 2 (GCN baseline ~55 Hits@50).
"""

_FILE = "pytorch-geometric-lp/custom_linkpred.py"

_CONTENT = """\
class LinkPredictor(nn.Module):
    \"\"\"GCN encoder + MLP decoder with pairwise features.

    MLP decoder on [z_src, z_dst, z_src*z_dst] matches OGB's standard GCN
    link-prediction baseline; strictly stronger than pure dot product.
    \"\"\"
    def __init__(self, in_channels: int, hidden_channels: int = 256,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers - 1)
        ])

        # MLP decoder on concatenated pair features
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def decode(self, edge_label_index: torch.Tensor, z: torch.Tensor,
               edge_index: Optional[torch.Tensor] = None,
               num_nodes: Optional[int] = None) -> torch.Tensor:
        z_src = z[edge_label_index[0]]
        z_dst = z[edge_label_index[1]]
        h = torch.cat([z_src, z_dst, z_src * z_dst], dim=-1)
        return self.decoder(h).squeeze(-1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_label_index: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, edge_index)
        return self.decode(edge_label_index, z,
                           edge_index=edge_index, num_nodes=x.size(0))

"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 127,
        "end_line": 210,
        "content": _CONTENT,
    },
]
