"""PiFold baseline for ai4bio-protein-inverse-folding.

Reference: vendor/external_packages/ProteinInvBench/PInvBench/src/models/pifold_model.py
Paper: Gao et al., "PiFold: Toward effective and efficient protein inverse folding" (ICLR 2023)

PiFold key innovations:
- Rich geometric features: multi-atom-pair RBF distances (Ca-Ca, N-N, C-C, O-O, Cb-Cb, etc.),
  dihedral angles, and local frame orientations for both nodes and edges
- Virtual atoms: learned positions in the local backbone frame to capture sidechain info
- Attention-based message passing with edge updates (NeighborAttention + EdgeMLP)
- Non-autoregressive MLP decoder
"""

_FILE = "ProteinInvBench/custom_invfold.py"

_CONTENT = """\
# =====================================================================
# EDITABLE SECTION START — PiFold baseline
# =====================================================================

import numpy as np


def gather_nodes_pifold(h_V, E_idx):
    \"\"\"Gather node features for neighbor nodes. Dense batched version.\"\"\"
    B, L, K = int(E_idx.shape[0]), int(E_idx.shape[1]), int(E_idx.shape[2])
    D = int(h_V.shape[-1])
    h_V_expand = h_V.unsqueeze(2).expand(-1, -1, K, -1)
    E_idx_expand = E_idx.unsqueeze(-1).expand(-1, -1, -1, D)
    return torch.gather(h_V_expand, 1, E_idx_expand)


class PiFoldAttention(nn.Module):
    \"\"\"Attention-based message passing layer inspired by PiFold's NeighborAttention.\"\"\"

    def __init__(self, hidden_dim, edge_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.d_head = hidden_dim // num_heads

        # Value network: processes edge-concatenated features
        self.W_V = nn.Sequential(
            nn.Linear(edge_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Attention bias from node+edge features
        self.Bias = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads),
        )
        self.W_O = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, h_V, h_E, E_idx, mask, mask_attend):
        \"\"\"
        h_V: (B, L, D), h_E: (B, L, K, D_e), E_idx: (B, L, K), mask: (B, L)
        \"\"\"
        B, L, K = int(E_idx.shape[0]), int(E_idx.shape[1]), int(E_idx.shape[2])
        D = self.hidden_dim
        n_heads = self.num_heads
        d = self.d_head

        # Gather neighbor features
        h_V_neighbors = gather_nodes_pifold(h_V, E_idx)  # (B, L, K, D)
        h_V_expand = h_V.unsqueeze(2).expand(-1, -1, K, -1)  # (B, L, K, D)

        # Edge + neighbor concatenation for value
        val_input = torch.cat([h_E, h_V_neighbors], dim=-1)  # (B, L, K, D_e+D)
        V = self.W_V(val_input).view(B, L, K, n_heads, d)  # (B, L, K, H, d)

        # Attention logits
        bias_input = torch.cat([h_V_expand, h_E, h_V_neighbors], dim=-1)
        w = self.Bias(bias_input).view(B, L, K, n_heads, 1) / np.sqrt(d)

        # Mask and softmax
        if mask_attend is not None:
            w = w + (1.0 - mask_attend.unsqueeze(-1).unsqueeze(-1)) * (-1e9)
        attend = torch.softmax(w, dim=2)  # (B, L, K, H, 1)

        # Aggregate
        h_V_update = (attend * V).sum(dim=2).reshape(B, L, D)  # (B, L, D)
        h_V_update = self.W_O(h_V_update)
        return h_V_update


class PiFoldEdgeMLP(nn.Module):
    \"\"\"Edge update network from PiFold.\"\"\"

    def __init__(self, hidden_dim, edge_dim, dropout=0.1):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim + edge_dim + hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.W3 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_V, h_E, E_idx, mask):
        B, L, K = int(E_idx.shape[0]), int(E_idx.shape[1]), int(E_idx.shape[2])
        h_V_neighbors = gather_nodes_pifold(h_V, E_idx)  # (B, L, K, D)
        h_V_expand = h_V.unsqueeze(2).expand(-1, -1, K, -1)
        h_EV = torch.cat([h_V_expand, h_E, h_V_neighbors], dim=-1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        # Apply batch norm per-feature
        D_e = int(h_E.shape[-1])
        h_E_flat = h_E.reshape(-1, D_e)
        h_msg_flat = h_message.reshape(-1, D_e)
        h_E = self.norm(h_E_flat + self.dropout(h_msg_flat)).reshape(B, L, K, D_e)
        return h_E


class PiFoldEncoderLayer(nn.Module):
    \"\"\"PiFold encoder layer: attention + FFN + edge update + context gating.\"\"\"

    def __init__(self, hidden_dim, edge_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = PiFoldAttention(hidden_dim, edge_dim, num_heads, dropout)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.edge_update = PiFoldEdgeMLP(hidden_dim, hidden_dim, dropout)
        # Context gating
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, h_V, h_E, E_idx, mask, mask_attend):
        B, L = int(h_V.shape[0]), int(h_V.shape[1])
        # Node update via attention
        D = int(h_V.shape[-1])
        dh = self.attention(h_V, h_E, E_idx, mask, mask_attend)
        h_V_flat = h_V.reshape(-1, D)
        dh_flat = dh.reshape(-1, D)
        h_V = self.norm1(h_V_flat + self.dropout(dh_flat)).reshape(B, L, -1)

        dh = self.ffn(h_V)
        h_V_flat = h_V.reshape(-1, D)
        dh_flat = dh.reshape(-1, D)
        h_V = self.norm2(h_V_flat + self.dropout(dh_flat)).reshape(B, L, -1)

        # Edge update
        h_E = self.edge_update(h_V, h_E, E_idx, mask)

        # Context gating (global information)
        # Mean pool over valid residues for context
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        c_V = (h_V * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / mask_sum.unsqueeze(-1)  # (B, 1, D)
        gate = self.context_gate(c_V.expand_as(h_V))
        h_V = h_V * gate

        h_V = h_V * mask.unsqueeze(-1)
        return h_V, h_E


class StructureEncoder(nn.Module):
    \"\"\"PiFold-style structure encoder with rich geometric features.\"\"\"

    def __init__(self, hidden_dim=128, num_layers=10, k_neighbors=30, dropout=0.1, num_rbf=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.num_rbf = num_rbf

        # PiFold uses rich multi-atom-pair features
        # Node features: 6 intra-residue atom-pair RBFs + 6 dihedrals + 9 orientations
        # = 6*num_rbf + 6 + 9
        node_input_dim = 6 * num_rbf + 12
        # Edge features: 15 inter-residue atom-pair RBFs + 4 angles + 12 directions + 16 pos_enc
        edge_input_dim = 15 * num_rbf + 4 + 8 + 16

        # Virtual atoms (learned positions in local frame, like PiFold)
        prior_matrix = [
            [-0.58273431, 0.56802827, -0.54067466],
            [0.0, 0.83867057, -0.54463904],
            [0.01984028, -0.78380804, -0.54183614],
        ]
        self.virtual_atoms = nn.Parameter(torch.tensor(prior_matrix, dtype=torch.float32))
        n_virtual = 3
        # Add virtual atom pair distances to both node and edge features
        node_input_dim += n_virtual * (n_virtual - 1) * num_rbf  # virtual-virtual pairs
        edge_input_dim += n_virtual * num_rbf + n_virtual * (n_virtual - 1) * num_rbf

        self.node_embed = nn.Linear(node_input_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_input_dim, hidden_dim)
        self.norm_nodes = nn.BatchNorm1d(hidden_dim)
        self.norm_edges = nn.BatchNorm1d(hidden_dim)

        self.W_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.W_e = nn.Linear(hidden_dim, hidden_dim)

        self.layers = nn.ModuleList([
            PiFoldEncoderLayer(hidden_dim, hidden_dim, num_heads=4, dropout=dropout)
            for _ in range(num_layers)
        ])

        self._init_params()

    def _init_params(self):
        for name, p in self.named_parameters():
            if name == 'virtual_atoms':
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _compute_features(self, X, mask, E_idx):
        \"\"\"Compute PiFold-style rich geometric features.\"\"\"
        B, L = int(X.shape[0]), int(X.shape[1])
        K = int(E_idx.shape[2])

        N_pos = X[:, :, 0, :]
        CA_pos = X[:, :, 1, :]
        C_pos = X[:, :, 2, :]
        O_pos = X[:, :, 3, :]

        # Virtual Cb and virtual atoms
        b = CA_pos - N_pos
        c = C_pos - CA_pos
        a = torch.cross(b, c, dim=-1)

        va = self.virtual_atoms / torch.norm(self.virtual_atoms, dim=1, keepdim=True)
        virtual_pos = []
        for i in range(int(va.shape[0])):
            vp = va[i, 0] * a + va[i, 1] * b + va[i, 2] * c + CA_pos
            virtual_pos.append(vp)

        # --- Node features ---
        def _node_rbf(_src, _dst):
            D = torch.sqrt(((_src - _dst) ** 2).sum(-1) + 1e-6)  # (B, L)
            return _rbf(D.unsqueeze(2), device=X.device).squeeze(2)  # (B, L, num_rbf)

        node_dist = []
        for _src, _dst in [(CA_pos, N_pos), (CA_pos, C_pos), (CA_pos, O_pos),
                      (N_pos, C_pos), (N_pos, O_pos), (O_pos, C_pos)]:
            node_dist.append(_node_rbf(_src, _dst))

        # Virtual atom pair distances (node-level)
        for i in range(len(virtual_pos)):
            for j in range(i):
                node_dist.append(_node_rbf(virtual_pos[i], virtual_pos[j]))
                node_dist.append(_node_rbf(virtual_pos[j], virtual_pos[i]))

        V_dist = torch.cat(node_dist, dim=-1)

        # Dihedrals and orientations
        V_dihedrals = _dihedrals(X)  # (B, L, 6)
        V_orient = _orientations(X)  # (B, L, 6)

        node_feat = torch.cat([V_dist, V_dihedrals, V_orient], dim=-1)

        # --- Edge features ---
        def _edge_rbf(_src, _dst, E_idx):
            D = torch.sqrt(((_src[:, :, None, :] - _dst[:, None, :, :]) ** 2).sum(-1) + 1e-6)
            D_neighbors = torch.gather(D, 2, E_idx)
            return _rbf(D_neighbors, device=X.device)

        edge_dist = []
        atom_pairs = [
            (CA_pos, CA_pos), (CA_pos, C_pos), (C_pos, CA_pos),
            (CA_pos, N_pos), (N_pos, CA_pos), (CA_pos, O_pos), (O_pos, CA_pos),
            (C_pos, C_pos), (C_pos, N_pos), (N_pos, C_pos),
            (C_pos, O_pos), (O_pos, C_pos), (N_pos, N_pos),
            (N_pos, O_pos), (O_pos, O_pos),
        ]
        for _src, _dst in atom_pairs:
            edge_dist.append(_edge_rbf(_src, _dst, E_idx))

        # Virtual atom edge features
        for i in range(len(virtual_pos)):
            edge_dist.append(_edge_rbf(virtual_pos[i], virtual_pos[i], E_idx))
            for j in range(i):
                edge_dist.append(_edge_rbf(virtual_pos[i], virtual_pos[j], E_idx))
                edge_dist.append(_edge_rbf(virtual_pos[j], virtual_pos[i], E_idx))

        E_dist = torch.cat(edge_dist, dim=-1)

        # Edge angles and directions
        CA_neighbors = gather_nodes_pifold(CA_pos, E_idx)  # (B, L, K, 3)
        dX = CA_neighbors - CA_pos.unsqueeze(2)
        dU = F.normalize(dX, dim=-1)

        fwd = F.normalize(CA_pos[:, 1:, :] - CA_pos[:, :-1, :], dim=-1)
        fwd = F.pad(fwd, (0, 0, 0, 1))
        n_vec = F.normalize(N_pos - CA_pos, dim=-1)
        c_vec = F.normalize(C_pos - CA_pos, dim=-1)
        o_vec = F.normalize(O_pos - CA_pos, dim=-1)

        # Direction features
        E_direct = torch.cat([
            (fwd.unsqueeze(2) * dU).sum(-1, keepdim=True),
            (n_vec.unsqueeze(2) * dU).sum(-1, keepdim=True),
            (c_vec.unsqueeze(2) * dU).sum(-1, keepdim=True),
            (o_vec.unsqueeze(2) * dU).sum(-1, keepdim=True),
            torch.cross(fwd.unsqueeze(2).expand_as(dU), dU, dim=-1).norm(dim=-1, keepdim=True),
            torch.cross(n_vec.unsqueeze(2).expand_as(dU), dU, dim=-1).norm(dim=-1, keepdim=True),
            torch.cross(c_vec.unsqueeze(2).expand_as(dU), dU, dim=-1).norm(dim=-1, keepdim=True),
            torch.cross(o_vec.unsqueeze(2).expand_as(dU), dU, dim=-1).norm(dim=-1, keepdim=True),
        ], dim=-1)  # (B, L, K, 8)

        # Edge angles (4): dihedral-like between consecutive neighbors
        E_angles = torch.cat([
            (dU[:, :, :, 0:1] * dU[:, :, :, 1:2]).clamp(-1, 1),
            (dU[:, :, :, 0:1] * dU[:, :, :, 2:3]).clamp(-1, 1),
            dU.norm(dim=-1, keepdim=True),
            dX.norm(dim=-1, keepdim=True) / 20.0,
        ], dim=-1)  # (B, L, K, 4)

        # Positional encoding
        residue_idx = torch.arange(L, device=X.device).unsqueeze(0).expand(B, -1)
        offset = residue_idx.unsqueeze(2) - torch.gather(
            residue_idx.unsqueeze(2).expand(-1, -1, K), 1,
            E_idx.clamp(0, L - 1)
        )
        pe_dim = 16
        freq = torch.exp(torch.arange(0, pe_dim, 2, dtype=torch.float32, device=X.device) * -(np.log(10000.0) / pe_dim))
        angles = offset.unsqueeze(-1).float() * freq
        pos_enc = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

        edge_feat = torch.cat([E_dist, E_angles, E_direct, pos_enc], dim=-1)

        return node_feat, edge_feat

    def forward(self, X, mask):
        B, L = int(X.shape[0]), int(X.shape[1])
        X_ca = X[:, :, 1, :]
        E_idx, _ = knn_graph(X_ca, mask, self.k_neighbors)
        K = int(E_idx.shape[2])

        # Compute features
        node_feat, edge_feat = self._compute_features(X, mask, E_idx)

        # Embed
        h_V_flat = self.node_embed(node_feat).reshape(-1, self.hidden_dim)
        h_V = self.norm_nodes(h_V_flat).reshape(B, L, self.hidden_dim)
        h_V = self.W_v[0](h_V)
        h_V_flat = h_V.reshape(-1, self.hidden_dim)
        h_V = self.W_v[2](self.W_v[1](h_V_flat)).reshape(B, L, self.hidden_dim)
        h_V = self.W_v[3](h_V)
        h_V_flat = h_V.reshape(-1, self.hidden_dim)
        h_V = self.W_v[5](self.W_v[4](h_V_flat)).reshape(B, L, self.hidden_dim)
        h_V = self.W_v[6](h_V)

        h_E_flat = self.edge_embed(edge_feat).reshape(-1, self.hidden_dim)
        h_E = self.norm_edges(h_E_flat).reshape(B, L, K, self.hidden_dim)
        h_E = self.W_e(h_E)

        # Attention mask
        mask_attend = torch.gather(mask.unsqueeze(2).expand(-1, -1, K), 1,
                                    E_idx.clamp(0, L - 1))
        mask_attend = mask.unsqueeze(-1) * mask_attend

        # Message passing
        for layer in self.layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        return h_V


class InverseFoldingModel(nn.Module):
    \"\"\"PiFold inverse folding model with non-autoregressive MLP decoder.\"\"\"

    def __init__(self, hidden_dim=128, num_encoder_layers=10, k_neighbors=30,
                 dropout=0.1, num_rbf=16):
        super().__init__()
        self.encoder = StructureEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            k_neighbors=k_neighbors,
            dropout=dropout,
            num_rbf=num_rbf,
        )
        self.decoder = nn.Linear(hidden_dim, NUM_AA)

    def forward(self, X, mask):
        h_V = self.encoder(X, mask)
        logits = self.decoder(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
"""

_CONFIG_OVERRIDE_CONTENT = """\
    CONFIG_OVERRIDES = {'num_encoder_layers': 10, 'batch_size': 8}
"""

# Ops are applied in order. Perform the late-file CONFIG_OVERRIDES replace
# first so the early editable block [86, 238] keeps its original line numbers.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 403,
        "end_line": 403,
        "content": _CONFIG_OVERRIDE_CONTENT,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 86,
        "end_line": 238,
        "content": _CONTENT,
    },
]
