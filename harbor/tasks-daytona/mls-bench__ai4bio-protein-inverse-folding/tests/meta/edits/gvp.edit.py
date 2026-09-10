"""GVP baseline for ai4bio-protein-inverse-folding.

Reference: vendor/external_packages/ProteinInvBench/PInvBench/src/models/gvp_model.py
Paper: Jing et al., "Learning from Protein Structure with Geometric Vector Perceptrons" (ICLR 2021)

GVP key innovations:
- Geometric Vector Perceptron: processes both scalar and vector features while
  maintaining SE(3) equivariance for vectors
- Node features: (6 scalar, 3 vector) from backbone geometry
- Edge features: (32 scalar, 1 vector) from RBF distances + direction vectors
- GVPConv layers with separate scalar/vector message passing
- Autoregressive decoder with encoder embedding injection
"""

_FILE = "ProteinInvBench/custom_invfold.py"

_CONTENT = """\
# =====================================================================
# EDITABLE SECTION START — GVP baseline
# =====================================================================

import numpy as np


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    \"\"\"L2 norm clamped above eps.\"\"\"
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def gather_nodes_gvp(h_V, E_idx):
    B, L, K = E_idx.shape
    D = h_V.shape[-1]
    h_V_expand = h_V.unsqueeze(2).expand(-1, -1, K, -1)
    E_idx_expand = E_idx.unsqueeze(-1).expand(-1, -1, -1, D)
    return torch.gather(h_V_expand, 1, E_idx_expand)


def gather_vectors(V, E_idx):
    \"\"\"Gather vector features. V: (B, L, n_vec, 3), E_idx: (B, L, K)\"\"\"
    B, L, K = E_idx.shape
    nv = V.shape[2]
    E_idx_v = E_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, nv, 3)
    V_expand = V.unsqueeze(2).expand(-1, -1, K, -1, -1)
    return torch.gather(V_expand, 1, E_idx_v)


class GVPModule(nn.Module):
    \"\"\"Geometric Vector Perceptron — dense batched version.

    Processes tuples of (scalar, vector) features.
    Scalar: (B, L, s_in) -> (B, L, s_out)
    Vector: (B, L, v_in, 3) -> (B, L, v_out, 3)
    \"\"\"

    def __init__(self, in_dims, out_dims, activations=(F.relu, torch.sigmoid)):
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.scalar_act, self.vector_act = activations

        if self.vi:
            self.h_dim = max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)

    def forward(self, s, v=None):
        if self.vi and v is not None:
            # v: (*, vi, 3)
            v_t = v.transpose(-1, -2)  # (*, 3, vi)
            vh = self.wh(v_t)  # (*, 3, h_dim)
            vn = _norm_no_nan(vh, axis=-2)  # (*, h_dim)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo:
                v_out = self.wv(vh).transpose(-1, -2)  # (*, vo, 3)
                if self.scalar_act:
                    gate = self.wsv(self.scalar_act(s))
                else:
                    gate = self.wsv(s)
                v_out = v_out * torch.sigmoid(gate).unsqueeze(-1)
            else:
                v_out = None
        else:
            s = self.ws(s)
            v_out = None

        if self.scalar_act:
            s = self.scalar_act(s)
        return s, v_out


class GVPLayerNorm(nn.Module):
    \"\"\"LayerNorm for GVP scalar+vector tuples.\"\"\"
    def __init__(self, dims):
        super().__init__()
        self.s_dim, self.v_dim = dims
        self.norm_s = nn.LayerNorm(self.s_dim)

    def forward(self, s, v=None):
        s = self.norm_s(s)
        if v is not None and self.v_dim > 0:
            vn = _norm_no_nan(v, axis=-1, keepdims=True)
            v = v / vn.clamp(min=1e-5)  # unit vectors, scaled
            v = v * vn  # restore magnitude (still normalized in mean)
        return s, v


class GVPConvLayer(nn.Module):
    \"\"\"GVP convolution layer — dense batched version.

    Message passing with GVP for both node and edge updates.
    \"\"\"

    def __init__(self, node_dims, edge_dims, drop_rate=0.1):
        super().__init__()
        self.node_s, self.node_v = node_dims
        self.edge_s, self.edge_v = edge_dims

        # Message function: edge_s + 2*node_s, edge_v + 2*node_v -> node_s, node_v
        msg_in_s = self.edge_s + 2 * self.node_s
        msg_in_v = self.edge_v + 2 * self.node_v
        self.msg_gvp = nn.Sequential(
            GVPModule((msg_in_s, msg_in_v), (self.node_s, self.node_v)),
            GVPModule((self.node_s, self.node_v), (self.node_s, self.node_v),
                      activations=(None, None)),
        )

        # Node update
        self.ff_gvp = nn.Sequential(
            GVPModule((self.node_s, self.node_v), (self.node_s * 4, self.node_v)),
            GVPModule((self.node_s * 4, self.node_v), (self.node_s, self.node_v),
                      activations=(None, None)),
        )

        self.norm1 = GVPLayerNorm(node_dims)
        self.norm2 = GVPLayerNorm(node_dims)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, h_s, h_v, e_s, e_v, E_idx, mask, mask_attend):
        \"\"\"
        h_s: (B, L, node_s), h_v: (B, L, node_v, 3)
        e_s: (B, L, K, edge_s), e_v: (B, L, K, edge_v, 3)
        E_idx: (B, L, K), mask: (B, L), mask_attend: (B, L, K)
        \"\"\"
        B, L, K = E_idx.shape

        # Gather neighbor node features
        h_s_j = gather_nodes_gvp(h_s, E_idx)  # (B, L, K, node_s)
        h_s_i = h_s.unsqueeze(2).expand(-1, -1, K, -1)

        # Build message input (scalar)
        msg_s = torch.cat([h_s_i, e_s, h_s_j], dim=-1)  # (B, L, K, msg_in_s)

        # Build message input (vector)
        if h_v is not None:
            h_v_j = gather_vectors(h_v, E_idx)  # (B, L, K, node_v, 3)
            h_v_i = h_v.unsqueeze(2).expand(-1, -1, K, -1, -1)
            if e_v is not None:
                msg_v = torch.cat([h_v_i, e_v, h_v_j], dim=-2)  # (B, L, K, msg_in_v, 3)
            else:
                msg_v = torch.cat([h_v_i, h_v_j], dim=-2)
        else:
            msg_v = e_v

        # Apply message GVP
        for layer in self.msg_gvp:
            msg_s, msg_v = layer(msg_s, msg_v)

        # Mask and aggregate
        mask_expand = mask_attend.unsqueeze(-1)
        msg_s = msg_s * mask_expand
        if msg_v is not None:
            msg_v = msg_v * mask_expand.unsqueeze(-1)

        # Sum aggregation
        num_neighbors = mask_attend.sum(dim=-1, keepdim=True).clamp(min=1)
        agg_s = msg_s.sum(dim=2) / num_neighbors
        if msg_v is not None:
            agg_v = msg_v.sum(dim=2) / num_neighbors.unsqueeze(-1)
        else:
            agg_v = None

        # Residual + norm
        h_s_res, h_v_res = self.norm1(h_s + self.drop(agg_s),
                                        h_v + self.drop(agg_v) if h_v is not None and agg_v is not None else h_v)

        # Feed-forward
        ff_s, ff_v = h_s_res, h_v_res
        for layer in self.ff_gvp:
            ff_s, ff_v = layer(ff_s, ff_v)

        h_s_out, h_v_out = self.norm2(h_s_res + self.drop(ff_s),
                                        h_v_res + self.drop(ff_v) if h_v_res is not None and ff_v is not None else h_v_res)

        # Mask
        h_s_out = h_s_out * mask.unsqueeze(-1)
        if h_v_out is not None:
            h_v_out = h_v_out * mask.unsqueeze(-1).unsqueeze(-1)

        return h_s_out, h_v_out


class StructureEncoder(nn.Module):
    \"\"\"GVP-based structure encoder.

    Uses geometric vector perceptrons for SE(3)-equivariant message passing.
    Node features: scalar (6) = dihedrals; vector (3) = local frame vectors.
    Edge features: scalar (32) = RBF distances + positional; vector (1) = direction.
    \"\"\"

    def __init__(self, hidden_dim=128, num_layers=3, k_neighbors=30, dropout=0.1, num_rbf=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.num_rbf = num_rbf

        # Dimensions
        self.node_s_in = 6    # dihedral sin/cos
        self.node_v_in = 3    # 3 direction vectors
        self.node_s_h = 100   # hidden scalar dim (GVP default)
        self.node_v_h = 16    # hidden vector dim
        self.edge_s_in = num_rbf + 16  # RBF + positional encoding
        self.edge_v_in = 1    # direction unit vector
        self.edge_s_h = 32    # hidden edge scalar
        self.edge_v_h = 1     # hidden edge vector

        # Input projections
        self.W_v = GVPModule(
            (self.node_s_in, self.node_v_in),
            (self.node_s_h, self.node_v_h),
            activations=(None, None)
        )
        self.norm_v = GVPLayerNorm((self.node_s_h, self.node_v_h))

        self.W_e = GVPModule(
            (self.edge_s_in, self.edge_v_in),
            (self.edge_s_h, self.edge_v_h),
            activations=(None, None)
        )
        self.norm_e = GVPLayerNorm((self.edge_s_h, self.edge_v_h))

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            GVPConvLayer(
                (self.node_s_h, self.node_v_h),
                (self.edge_s_h, self.edge_v_h),
                drop_rate=dropout
            )
            for _ in range(num_layers)
        ])

        # Output projection to scalar hidden_dim
        self.out_proj = nn.Linear(self.node_s_h, hidden_dim)

    def forward(self, X, mask):
        B, L = int(X.shape[0]), int(X.shape[1])
        X_ca = X[:, :, 1, :]

        # Build KNN graph
        E_idx, D_neighbors = knn_graph(X_ca, mask, self.k_neighbors)
        K = int(E_idx.shape[2])

        # Node features
        # Scalar: dihedral angles
        node_s = _dihedrals(X)  # (B, L, 6)

        # Vector: local frame vectors (CA->N, CA->C, CA->O unit vectors)
        N_pos, CA_pos, C_pos, O_pos = X[:, :, 0], X[:, :, 1], X[:, :, 2], X[:, :, 3]
        v_cn = F.normalize(N_pos - CA_pos, dim=-1)   # (B, L, 3)
        v_cc = F.normalize(C_pos - CA_pos, dim=-1)
        v_co = F.normalize(O_pos - CA_pos, dim=-1)
        node_v = torch.stack([v_cn, v_cc, v_co], dim=2)  # (B, L, 3, 3)

        # Edge features
        # Scalar: RBF distances + positional encoding
        rbf = _rbf(D_neighbors, device=X.device)  # (B, L, K, num_rbf)
        residue_idx = torch.arange(L, device=X.device).unsqueeze(0).expand(B, -1)
        offset = residue_idx.unsqueeze(2) - torch.gather(
            residue_idx.unsqueeze(2).expand(-1, -1, K), 1,
            E_idx.clamp(0, L - 1)
        )
        pe_dim = 16
        freq = torch.exp(torch.arange(0, pe_dim, 2, dtype=torch.float32, device=X.device) * -(np.log(10000.0) / pe_dim))
        angles = offset.unsqueeze(-1).float() * freq
        pos_enc = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        edge_s = torch.cat([rbf, pos_enc], dim=-1)  # (B, L, K, num_rbf+16)

        # Vector: direction to neighbors
        CA_neighbors = gather_nodes_gvp(CA_pos, E_idx)  # (B, L, K, 3)
        edge_dir = F.normalize(CA_neighbors - CA_pos.unsqueeze(2), dim=-1)  # (B, L, K, 3)
        edge_v = edge_dir.unsqueeze(3)  # (B, L, K, 1, 3)

        # Project inputs
        h_s, h_v = self.W_v(node_s, node_v)
        h_s, h_v = self.norm_v(h_s, h_v)

        e_s, e_v = self.W_e(edge_s, edge_v)
        e_s, e_v = self.norm_e(e_s, e_v)

        # Attention mask
        mask_attend = torch.gather(mask.unsqueeze(2).expand(-1, -1, K), 1,
                                    E_idx.clamp(0, L - 1))
        mask_attend = mask.unsqueeze(-1) * mask_attend

        # Message passing
        for layer in self.encoder_layers:
            h_s, h_v = layer(h_s, h_v, e_s, e_v, E_idx, mask, mask_attend)

        # Project to output dim
        h_V = self.out_proj(h_s)
        return h_V


class InverseFoldingModel(nn.Module):
    \"\"\"GVP inverse folding model.\"\"\"

    def __init__(self, hidden_dim=128, num_encoder_layers=3, k_neighbors=30,
                 dropout=0.1, num_rbf=16):
        super().__init__()
        self.encoder = StructureEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            k_neighbors=k_neighbors,
            dropout=dropout,
            num_rbf=num_rbf,
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_AA),
        )

    def forward(self, X, mask):
        h_V = self.encoder(X, mask)
        logits = self.decoder(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 86,
        "end_line": 238,
        "content": _CONTENT,
    },
]
