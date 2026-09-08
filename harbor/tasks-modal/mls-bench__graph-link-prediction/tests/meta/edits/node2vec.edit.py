"""Node2Vec-inspired embedding baseline for graph-link-prediction.

Grover & Leskovec, "node2vec: Scalable Feature Learning for Networks", KDD 2016.
This benchmark variant learns per-node embeddings directly, combines them with
input features, and trains an MLP decoder for link prediction. It does not run
node2vec random walks.
"""

_FILE = "pytorch-geometric-lp/custom_linkpred.py"

_CONTENT = """\
class LinkPredictor(nn.Module):
    \"\"\"Node2Vec-style learnable node embeddings + GCN feature encoder + MLP decoder.

    Combines learnable per-node embeddings with a lightweight GCN that
    encodes input features. The embedding is allocated eagerly at a safe
    upper-bound size so that all parameters are visible to the optimizer
    from the start.

    The MLP decoder scores pairs of node embeddings (following SEAL-style
    pairwise features: concatenation + Hadamard product).
    \"\"\"
    def __init__(self, in_channels: int, hidden_channels: int = 256,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        # Eagerly allocate embedding for up to 250000 nodes (covers Planetoid
        # and ogbl-collab, which has 235868 nodes).  Only first num_nodes rows used.
        max_num_nodes = 250000
        self.node_emb = nn.Embedding(max_num_nodes, hidden_channels)
        nn.init.xavier_uniform_(self.node_emb.weight)

        # Lightweight feature encoder: single-layer linear projection
        self.feat_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # MLP decoder with pairwise features (concat + hadamard)
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
        num_nodes = x.size(0)
        node_ids = torch.arange(num_nodes, device=x.device)
        z = self.node_emb(node_ids)
        # Combine with projected input features
        z = z + self.feat_proj(x)
        return z

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
