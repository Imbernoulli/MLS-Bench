"""EGNN baseline for ai4bio-protein-structure-repr.

Ported DIRECTLY from ProteinWorkshop reference implementation:
  vendor/external_packages/ProteinWorkshop/proteinworkshop/models/graph_encoders/egnn.py
  vendor/external_packages/ProteinWorkshop/proteinworkshop/models/graph_encoders/layers/egnn.py

Uses E(n)-equivariant message passing that jointly updates node features
and coordinates.

Hyperparameters match the reference config (egnn.yaml):
  num_layers=6, emb_dim=512, activation=relu, norm=batch, aggr=sum,
  pool=mean, residual=True, dropout=0.1
"""

_FILE = "ProteinWorkshop/custom_protein_encoder.py"

_CONTENT = """\
# =====================================================================
# EDITABLE SECTION START — EGNN encoder (ported from ProteinWorkshop)
# =====================================================================

import torch_scatter
from torch.nn import Linear, Dropout, Sequential
from torch_geometric.nn import MessagePassing

class EGNNLayer(MessagePassing):
    \"\"\"E(n) Equivariant GNN Layer.

    Ported directly from ProteinWorkshop:
      proteinworkshop/models/graph_encoders/layers/egnn.py

    Paper: E(n) Equivariant Graph Neural Networks, Satorras et al. (ICML 2021)
    \"\"\"
    def __init__(self, emb_dim, activation='relu', norm='batch', aggr='sum', dropout=0.1):
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim

        # Normalization layer (matching reference)
        norm_cls = {
            'layer': nn.LayerNorm,
            'batch': nn.BatchNorm1d,
        }[norm]

        # Helper to create fresh activation instances
        def _make_act():
            if activation == 'relu':
                return nn.ReLU()
            elif activation in ('silu', 'swish'):
                return nn.SiLU()
            elif activation == 'elu':
                return nn.ELU()
            return nn.ReLU()

        # MLP psi_h for computing messages m_ij (matching reference exactly)
        self.mlp_msg = Sequential(
            Linear(2 * emb_dim + 1, emb_dim),
            norm_cls(emb_dim),
            _make_act(),
            Dropout(dropout),
            Linear(emb_dim, emb_dim),
            norm_cls(emb_dim),
            _make_act(),
            Dropout(dropout),
        )
        # MLP psi_x for computing coordinate displacement weights
        self.mlp_pos = Sequential(
            Linear(emb_dim, emb_dim),
            norm_cls(emb_dim),
            _make_act(),
            Dropout(dropout),
            Linear(emb_dim, 1),
        )
        # MLP phi for computing updated node features
        self.mlp_upd = Sequential(
            Linear(2 * emb_dim, emb_dim),
            norm_cls(emb_dim),
            _make_act(),
            Dropout(dropout),
            Linear(emb_dim, emb_dim),
            norm_cls(emb_dim),
            _make_act(),
            Dropout(dropout),
        )

    def forward(self, h, pos, edge_index):
        \"\"\"
        Args:
            h: (n, d) - initial node features
            pos: (n, 3) - initial node coordinates
            edge_index: (2, e) - edge indices
        Returns:
            msg_aggr: (n, d) - updated node features delta
            pos_aggr: (n, 3) - coordinate displacement
        \"\"\"
        msg_aggr, pos_aggr = self.propagate(edge_index, h=h, pos=pos)
        msg_aggr = self.mlp_upd(torch.cat([h, msg_aggr], dim=-1))
        return msg_aggr, pos_aggr

    def message(self, h_i, h_j, pos_i, pos_j):
        \"\"\"Compute messages (matching reference exactly).\"\"\"
        pos_diff = pos_i - pos_j
        dists = torch.norm(pos_diff, dim=-1, keepdim=True)
        msg = torch.cat([h_i, h_j, dists], dim=-1)
        msg = self.mlp_msg(msg)
        # Scale displacement vector by learned weight
        pos_diff = pos_diff / (dists + 1) * self.mlp_pos(msg)
        return msg, pos_diff

    def aggregate(self, inputs, index):
        \"\"\"Aggregate messages and position displacements separately (matching reference).\"\"\"
        msgs, pos_diffs = inputs
        # Aggregate messages using configured aggr (sum in reference config)
        msg_aggr = torch_scatter.scatter(
            msgs, index, dim=self.node_dim, reduce=self.aggr
        )
        # Aggregate displacement vectors always with mean (matching reference)
        pos_aggr = torch_scatter.scatter(
            pos_diffs, index, dim=self.node_dim, reduce="mean"
        )
        return msg_aggr, pos_aggr

    def __repr__(self):
        return f"{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})"


class ProteinEncoder(nn.Module):
    \"\"\"EGNN-based protein structure encoder.

    Ported directly from ProteinWorkshop EGNNModel.
    E(n)-equivariant: jointly updates node features and coordinates.
    Uses residual connections on both features and coordinates.

    Reference hyperparameters (from proteinworkshop/config/encoder/egnn.yaml):
      num_layers=6, emb_dim=512, activation=relu, norm=batch, aggr=sum,
      pool=mean, residual=True, dropout=0.1
    \"\"\"
    def __init__(
        self,
        input_dim: int = SCALAR_NODE_DIM,
        hidden_dim: int = 256,
        out_dim: int = 128,
        num_layers: int = 6,
        dropout: float = 0.1,
        cutoff: float = 10.0,
        max_neighbors: int = 16,
    ):
        super().__init__()
        # Override with ProteinWorkshop reference hyperparameters
        emb_dim = 512
        activation = 'relu'
        norm = 'batch'
        aggr = 'sum'
        residual = True

        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.residual = residual

        # Embedding lookup for initial node features (matching reference LazyLinear)
        self.emb_in = nn.Linear(input_dim, emb_dim)

        # Stack of EGNN layers (matching reference)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(EGNNLayer(emb_dim, activation, norm, aggr, dropout))

        # Global pooling/readout: mean (matching reference config)
        self.pool = global_mean_pool

        # Output projection to match expected out_dim
        self.out_proj = nn.Linear(emb_dim, out_dim)

    def _build_edges(self, pos, batch):
        \"\"\"Build kNN graph for message passing.\"\"\"
        edge_index = knn_graph(pos, k=self.max_neighbors, batch=batch, loop=False)
        return edge_index

    def forward(self, pos, node_feat, batch):
        \"\"\"Forward pass matching ProteinWorkshop EGNNModel.

        Args:
            pos: (N, 3) alpha-carbon coordinates
            node_feat: (N, input_dim) node scalar features
            batch: (N,) batch index

        Returns:
            node_emb: (N, out_dim) per-node embeddings
            graph_emb: (B, out_dim) per-graph embeddings
        \"\"\"
        edge_index = self._build_edges(pos, batch)

        h = self.emb_in(node_feat)  # (n, input_dim) -> (n, emb_dim)

        for conv in self.convs:
            # Message passing layer
            h_update, pos_update = conv(h, pos, edge_index)

            # Update node features with residual (matching reference)
            h = h + h_update if self.residual else h_update

            # Update node coordinates with residual (matching reference)
            pos = pos + pos_update if self.residual else pos_update

        # Project to output dimension
        node_emb = self.out_proj(h)
        graph_emb = self.pool(node_emb, batch)

        return node_emb, graph_emb

# =====================================================================
# EDITABLE SECTION END
# =====================================================================
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 125,
        "end_line": 252,
        "content": _CONTENT,
    },
]
