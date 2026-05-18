"""BUDDY baseline for graph-link-prediction.

Chamberlain et al., "Graph Neural Networks for Link Prediction with Subgraph
Sketching", ICLR 2023.

BUDDY uses feature precomputation (subgraph sketches via hashing) to capture
structural information without expensive subgraph extraction. Key idea:
precompute node-level structural features (common neighbor counts, hash-based
set sketches) and combine with GNN embeddings via an MLP decoder.

Reported: Cora AUC ~95.1, CiteSeer AUC ~96.7, ogbl-collab Hits@50 ~64.6
"""

_FILE = "pytorch-geometric-lp/custom_linkpred.py"

_CONTENT = """\
class StructuralFeatureComputer:
    \"\"\"Precomputes structural pairwise features (approximating BUDDY sketches).\"\"\"

    @staticmethod
    @torch.no_grad()
    def compute_cn_features(edge_index, num_nodes, edge_label_index):
        \"\"\"Compute CN/AA/RA features using scipy sparse (memory-efficient).\"\"\"
        import scipy.sparse as sp
        device = edge_label_index.device

        row = edge_index[0].cpu().numpy()
        col = edge_index[1].cpu().numpy()
        adj = sp.csr_matrix((np.ones(len(row)), (row, col)),
                            shape=(num_nodes, num_nodes))

        src = edge_label_index[0].cpu().numpy()
        dst = edge_label_index[1].cpu().numpy()

        # Sparse row extraction + element-wise multiply stays sparse
        src_rows = adj[src]   # [batch, N] sparse
        dst_rows = adj[dst]   # [batch, N] sparse
        common = src_rows.multiply(dst_rows)  # sparse intersection

        deg = np.asarray(adj.sum(axis=1)).flatten().clip(min=1)
        cn = np.asarray(common.sum(axis=1)).flatten()
        aa = np.asarray(common.multiply(1.0 / np.log(deg).clip(min=1.0))
                        .sum(axis=1)).flatten()
        ra = np.asarray(common.multiply(1.0 / deg).sum(axis=1)).flatten()

        return torch.tensor(np.stack([cn, aa, ra], axis=1),
                            dtype=torch.float32, device=device)


class LinkPredictor(nn.Module):
    \"\"\"BUDDY-inspired link predictor.

    Combines GCN node embeddings with precomputed structural features
    (common neighbors, Adamic-Adar, resource allocation) via an MLP decoder.
    This approximates BUDDY's subgraph sketching approach.

    The new decode interface takes `edge_label_index` (original node
    indices) and the full embedding table `z` directly, so we no longer
    need to recover indices via hashing/argmax.  The training graph
    `edge_index` is also passed through, enabling exact CN/AA/RA
    computation against whichever adjacency is in use (train-only during
    validation, train+val during final test, as OGB prescribes).
    \"\"\"
    def __init__(self, in_channels: int, hidden_channels: int = 256,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # GCN encoder
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])

        # Structural feature dimension: CN, AA, RA = 3
        struct_dim = 3
        self.struct_proj = nn.Linear(struct_dim, hidden_channels)

        # MLP decoder: node features + structural features
        dec_in = hidden_channels * 2 + hidden_channels  # src, dst, struct
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

        # Cached context set at encode-time so decode() has sensible
        # defaults when the caller does not pass edge_index explicitly.
        self._edge_index = None
        self._num_nodes = None

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        self._edge_index = edge_index
        self._num_nodes = x.size(0)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def decode(self, edge_label_index: torch.Tensor, z: torch.Tensor,
               edge_index: Optional[torch.Tensor] = None,
               num_nodes: Optional[int] = None) -> torch.Tensor:
        # Resolve the adjacency to use for structural features.
        ei = edge_index if edge_index is not None else self._edge_index
        N = num_nodes if num_nodes is not None else (
            self._num_nodes if self._num_nodes is not None else z.size(0))

        with torch.no_grad():
            struct_feats = StructuralFeatureComputer.compute_cn_features(
                ei, N, edge_label_index)
        struct_h = self.struct_proj(struct_feats.float())

        z_src = z[edge_label_index[0]]
        z_dst = z[edge_label_index[1]]
        h = torch.cat([z_src, z_dst, struct_h], dim=-1)
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
