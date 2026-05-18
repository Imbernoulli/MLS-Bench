"""ProteinMPNN baseline for ai4bio-protein-inverse-folding.

Reference: vendor/external_packages/ProteinInvBench/PInvBench/src/modules/proteinmpnn_module.py
Paper: Dauparas et al., "Robust deep learning-based protein sequence design using ProteinMPNN" (Science, 2022)

ProteinMPNN uses:
- Encoder: MPNN with node+edge updates, edge features from all-atom pairwise RBFs
- Decoder: MLP prediction head (simplified from the original autoregressive decoder)
- Key features: 25 pairwise distance RBFs (all backbone + Cb atoms), positional encodings
"""

_FILE = "ProteinInvBench/custom_invfold.py"

_CONTENT = """\
# =====================================================================
# EDITABLE SECTION START — ProteinMPNN baseline
# =====================================================================

import numpy as np

class ProteinFeatures(nn.Module):
    \"\"\"Extract protein structural features: all-atom pairwise RBFs + positional.

    Computes 25 pairwise RBF distance matrices between all backbone atom
    pairs (N, CA, C, O, Cb) following the reference ProteinMPNN implementation.
    Total edge features = 25 * num_rbf + num_pos_emb.
    \"\"\"
    def __init__(self, edge_features, node_features, num_pos_emb=16, num_rbf=16,
                 top_k=30, augment_eps=0.0):
        super().__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_pos_emb = num_pos_emb

        # 25 pairwise RBFs + positional encoding
        edge_in = num_pos_emb + num_rbf * 25
        node_in = 6  # forward + side-chain orientation vectors

        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)
        self.node_embedding = nn.Linear(node_in, node_features, bias=True)
        self.norm_nodes = nn.LayerNorm(node_features)

    def _pos_enc(self, E_idx):
        N_nodes = E_idx.size(1)
        ii = torch.arange(N_nodes, dtype=torch.float32, device=E_idx.device).view(1, -1, 1)
        d = (E_idx.float() - ii).unsqueeze(-1)
        frequency = torch.exp(
            torch.arange(0, self.num_pos_emb, 2, dtype=torch.float32, device=E_idx.device)
            * -(np.log(10000.0) / self.num_pos_emb)
        )
        angles = d * frequency.view(1, 1, 1, -1)
        return torch.cat([torch.cos(angles), torch.sin(angles)], -1)

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
        dX = X.unsqueeze(1) - X.unsqueeze(2)
        D = (1. - mask_2D) * 10000 + mask_2D * torch.sqrt((dX ** 2).sum(3) + eps)
        D_max, _ = D.max(-1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * (D_max + 1)
        D_neighbors, E_idx = torch.topk(D_adjust, min(self.top_k, D_adjust.shape[-1]),
                                         dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf_fn(self, D):
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device).view(1, 1, 1, -1)
        D_sigma = (D_max - D_min) / D_count
        return torch.exp(-((D.unsqueeze(-1) - D_mu) / D_sigma) ** 2)

    def _get_rbf(self, A, B, E_idx):
        \"\"\"Compute pairwise distances between atoms A and B, gather for neighbors, apply RBF.\"\"\"
        D_AB = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6)
        # Gather neighbor distances
        B_size, L, K = E_idx.shape
        E_idx_expand = E_idx.unsqueeze(-1)  # (B, L, K, 1)
        D_AB_expand = D_AB.unsqueeze(2).expand(-1, -1, K, -1)  # (B, L, K, L)
        # For each node i and neighbor j = E_idx[i,k], get D_AB[i, j]
        D_AB_neighbors = torch.gather(D_AB_expand, 3, E_idx_expand).squeeze(-1)  # (B, L, K)
        return self._rbf_fn(D_AB_neighbors)

    def _orientations(self, X):
        fwd = F.normalize(X[:, 1:, :] - X[:, :-1, :], dim=-1)
        fwd = F.pad(fwd, (0, 0, 0, 1))
        return fwd

    def _sidechains(self, X):
        n, ca, c = X[:, :, 0, :], X[:, :, 1, :], X[:, :, 2, :]
        u = F.normalize(n - ca, dim=-1)
        v = F.normalize(c - ca, dim=-1)
        return F.normalize(u - v, dim=-1)

    def forward(self, X, mask, residue_idx=None, chain_encoding=None):
        B, L = X.shape[0], X.shape[1]
        N = X[:, :, 0, :]   # N atoms
        Ca = X[:, :, 1, :]  # CA atoms
        C = X[:, :, 2, :]   # C atoms
        O = X[:, :, 3, :]   # O atoms

        # Virtual Cb (beta carbon from N-CA-C geometry)
        b = N - Ca
        c = C - Ca
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

        # KNN based on CA distances
        D_neighbors, E_idx = self._dist(Ca, mask)

        # All 25 pairwise RBF distances (matching reference ProteinMPNN)
        RBF_all = []
        RBF_all.append(self._rbf_fn(D_neighbors))  # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))
        RBF_all.append(self._get_rbf(C, C, E_idx))
        RBF_all.append(self._get_rbf(O, O, E_idx))
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))
        RBF_all.append(self._get_rbf(Ca, N, E_idx))
        RBF_all.append(self._get_rbf(Ca, C, E_idx))
        RBF_all.append(self._get_rbf(Ca, O, E_idx))
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))
        RBF_all.append(self._get_rbf(N, C, E_idx))
        RBF_all.append(self._get_rbf(N, O, E_idx))
        RBF_all.append(self._get_rbf(N, Cb, E_idx))
        RBF_all.append(self._get_rbf(Cb, C, E_idx))
        RBF_all.append(self._get_rbf(Cb, O, E_idx))
        RBF_all.append(self._get_rbf(O, C, E_idx))
        RBF_all.append(self._get_rbf(N, Ca, E_idx))
        RBF_all.append(self._get_rbf(C, Ca, E_idx))
        RBF_all.append(self._get_rbf(O, Ca, E_idx))
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))
        RBF_all.append(self._get_rbf(C, N, E_idx))
        RBF_all.append(self._get_rbf(O, N, E_idx))
        RBF_all.append(self._get_rbf(Cb, N, E_idx))
        RBF_all.append(self._get_rbf(C, Cb, E_idx))
        RBF_all.append(self._get_rbf(O, Cb, E_idx))
        RBF_all.append(self._get_rbf(C, O, E_idx))
        RBF_all = torch.cat(RBF_all, dim=-1)  # (B, L, K, 25*num_rbf)

        # Positional encoding
        O_pos = self._pos_enc(E_idx)  # (B, L, K, num_pos_emb)

        # Edge features: positional + all-atom RBFs
        E = torch.cat([O_pos, RBF_all], dim=-1)

        # Node features: forward + side-chain orientation vectors
        O_fwd = self._orientations(Ca)
        O_sc = self._sidechains(X)
        V = torch.cat([O_fwd, O_sc], dim=-1)

        V = self.norm_nodes(self.node_embedding(V))
        E = self.norm_edges(self.edge_embedding(E))
        return V, E, E_idx


def gather_nodes(h_V, E_idx):
    \"\"\"Gather node features for neighbor nodes.\"\"\"
    B, L, K = E_idx.shape
    D = h_V.shape[-1]
    h_V_expand = h_V.unsqueeze(2).expand(-1, -1, K, -1)
    E_idx_expand = E_idx.unsqueeze(-1).expand(-1, -1, -1, D)
    return torch.gather(h_V_expand, 1, E_idx_expand)


def cat_neighbors_nodes(h_nodes, h_edges, E_idx):
    \"\"\"Concatenate neighbor node features with edge features.\"\"\"
    h_V_neighbors = gather_nodes(h_nodes, E_idx)
    return torch.cat([h_edges, h_V_neighbors], dim=-1)


class EncLayer(nn.Module):
    \"\"\"ProteinMPNN encoder layer with node and edge updates.\"\"\"
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super().__init__()
        self.num_hidden = num_hidden
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden)
        self.W2 = nn.Linear(num_hidden, num_hidden)
        self.W3 = nn.Linear(num_hidden, num_hidden)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden)
        self.W12 = nn.Linear(num_hidden, num_hidden)
        self.W13 = nn.Linear(num_hidden, num_hidden)
        self.act = nn.GELU()
        self.dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden * 4),
            nn.GELU(),
            nn.Linear(num_hidden * 4, num_hidden),
        )

    def forward(self, h_V, h_E, E_idx, mask, mask_attend):
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = h_message.sum(-2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask is not None:
            h_V = mask.unsqueeze(-1) * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class StructureEncoder(nn.Module):
    \"\"\"ProteinMPNN-style structure encoder with all-atom pairwise features.\"\"\"

    def __init__(self, hidden_dim=128, num_layers=3, k_neighbors=30, dropout=0.1, num_rbf=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors

        self.features = ProteinFeatures(
            hidden_dim, hidden_dim, top_k=k_neighbors, augment_eps=0.0, num_rbf=num_rbf
        )
        self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_layers)
        ])

        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, mask):
        V, E, E_idx = self.features(X, mask)

        # Start with zero node features (per reference ProteinMPNN)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        return h_V


class InverseFoldingModel(nn.Module):
    \"\"\"ProteinMPNN inverse folding model.\"\"\"

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
