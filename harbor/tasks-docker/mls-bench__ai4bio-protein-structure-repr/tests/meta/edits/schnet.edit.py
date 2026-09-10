"""SchNet baseline for ai4bio-protein-structure-repr.

Ported DIRECTLY from ProteinWorkshop reference implementation:
  vendor/external_packages/ProteinWorkshop/proteinworkshop/models/graph_encoders/schnet.py

Uses PyG's SchNet InteractionBlock (CFConv + ShiftedSoftplus + Linear).
Hyperparameters match the reference config (schnet.yaml):
  hidden_channels=512, num_filters=128, num_gaussians=50, cutoff=10.0,
  max_num_neighbors=32, readout="add".
"""

_FILE = "ProteinWorkshop/custom_protein_encoder.py"

_CONTENT = """\
# =====================================================================
# EDITABLE SECTION START — SchNet encoder (ported from ProteinWorkshop)
# =====================================================================

# Import PyG SchNet components used by the reference implementation
from torch_geometric.nn.models.schnet import InteractionBlock, GaussianSmearing, ShiftedSoftplus

class ProteinEncoder(nn.Module):
    \"\"\"SchNet-based protein structure encoder.

    Ported directly from ProteinWorkshop SchNetModel.
    Uses continuous-filter convolutions with Gaussian RBF distance expansion.
    Invariant to rotations and translations by design.

    Reference hyperparameters (from proteinworkshop/config/encoder/schnet.yaml):
      hidden_channels=512, num_filters=128, num_gaussians=50, cutoff=10.0,
      max_num_neighbors=32, readout="add"
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
        hidden_channels = 512
        num_filters = 128
        num_gaussians = 50
        self.cutoff = cutoff
        max_num_neighbors = 32
        readout = "add"

        self.hidden_channels = hidden_channels
        self.out_dim = out_dim
        self.max_num_neighbors = max_num_neighbors
        self.readout = readout

        # Overwrite embedding to accept arbitrary input features (matching reference LazyLinear)
        self.embedding = nn.Linear(input_dim, hidden_channels)

        # Gaussian RBF distance expansion (from PyG SchNet)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        # Stack of InteractionBlocks (from PyG SchNet)
        self.interactions = nn.ModuleList()
        for _ in range(num_layers):
            block = InteractionBlock(
                hidden_channels, num_gaussians, num_filters, cutoff
            )
            self.interactions.append(block)

        # Output MLP: lin1 -> act -> lin2 (matching reference)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.act = ShiftedSoftplus()
        self.lin2 = nn.Linear(hidden_channels, out_dim)

    def _build_edges(self, pos, batch):
        \"\"\"Build kNN graph and compute edge weights + RBF features.\"\"\"
        edge_index = knn_graph(
            pos, k=self.max_num_neighbors, batch=batch, loop=False
        )
        u, v = edge_index
        edge_weight = (pos[u] - pos[v]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)
        return edge_index, edge_weight, edge_attr

    def forward(self, pos, node_feat, batch):
        \"\"\"Forward pass matching ProteinWorkshop SchNetModel.

        Args:
            pos: (N, 3) alpha-carbon coordinates
            node_feat: (N, input_dim) node scalar features
            batch: (N,) batch index

        Returns:
            node_emb: (N, out_dim) per-node embeddings
            graph_emb: (B, out_dim) per-graph embeddings
        \"\"\"
        edge_index, edge_weight, edge_attr = self._build_edges(pos, batch)

        # Project input features to hidden dimension
        h = self.embedding(node_feat)

        # Message passing with residual connections (matching reference exactly)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        # Output projection: lin1 -> act -> lin2 (matching reference)
        h = self.lin1(h)
        h = self.act(h)
        node_emb = self.lin2(h)

        # Graph-level readout via scatter (matching reference readout="add")
        graph_emb = scatter_add(node_emb, batch, dim=0)

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
