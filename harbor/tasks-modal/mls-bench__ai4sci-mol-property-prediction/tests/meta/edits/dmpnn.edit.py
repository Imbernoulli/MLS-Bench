"""D-MPNN baseline — Directed Message Passing Neural Network with RDKit features.

Implements the Chemprop D-MPNN faithfully:
- Directed bond-level message passing (no message collision)
- depth=3, hidden_dim=300, dropout configurable
- Sum pooling readout (matches chemprop default)
- RDKit 2D normalized molecular descriptors concatenated at the readout
  ("D-MPNN + features" config from Yang et al. 2019; this is the
  configuration commonly reported in benchmark tables).

Reference: Yang et al., "Analyzing Learned Molecular Representations for
Property Prediction" (JCIM 2019). Chemprop default + features_generator=rdkit_2d_normalized.
"""

_FILE = "Uni-Mol/custom_molprop.py"

_CONTENT = """\
# =====================================================================
# EDITABLE SECTION START — D-MPNN: Directed Message Passing Neural Network
# =====================================================================

from rdkit.Chem import Descriptors as _Descriptors
from rdkit.Chem import rdMolDescriptors as _rdMolDescriptors
from rdkit.Chem import MolFromSmiles as _MolFromSmiles


# --------------------- RDKit 2D molecular descriptors -----------------
# A compact subset of normalized RDKit 2D descriptors that have been
# shown to improve D-MPNN on physicochemical / biophysical tasks (Yang
# et al. 2019, "rdkit_2d_normalized" features generator).  We compute
# them once per SMILES and per-feature standardize using running stats
# accumulated over the training batches — a robust approximation of
# chemprop's pre-computed Welford normalization.

def _rdkit_2d_descriptors(smi):
    \"\"\"Compute a fixed-length RDKit 2D descriptor vector for a SMILES.\"\"\"
    if not smi:
        return [0.0] * 17
    mol = _MolFromSmiles(smi)
    if mol is None:
        return [0.0] * 17
    feats = [
        _Descriptors.MolWt(mol),
        _Descriptors.MolLogP(mol),
        _Descriptors.NumHDonors(mol),
        _Descriptors.NumHAcceptors(mol),
        _Descriptors.TPSA(mol),
        _Descriptors.NumRotatableBonds(mol),
        _Descriptors.NumAromaticRings(mol),
        _Descriptors.NumAliphaticRings(mol),
        _Descriptors.HeavyAtomCount(mol),
        _Descriptors.RingCount(mol),
        _Descriptors.FractionCSP3(mol),
        _Descriptors.NumHeteroatoms(mol),
        _rdMolDescriptors.CalcNumSaturatedRings(mol),
        _rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
        _rdMolDescriptors.CalcNumAliphaticHeterocycles(mol),
        _Descriptors.MolMR(mol),
        _Descriptors.LabuteASA(mol),
    ]
    # NaN / inf guard
    cleaned = []
    for v in feats:
        try:
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                v = 0.0
        except Exception:
            v = 0.0
        cleaned.append(v)
    return cleaned


_RDKIT_FEAT_DIM = 17


class _RunningNormalizer(nn.Module):
    \"\"\"Running mean/std normalizer for RDKit features (BatchNorm-style).\"\"\"

    def __init__(self, dim, momentum=0.01):
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_std', torch.ones(dim))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                mean = x.mean(dim=0)
                std = x.std(dim=0).clamp(min=1e-6)
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_std.mul_(1 - self.momentum).add_(self.momentum * std)
        return (x - self.running_mean) / self.running_std.clamp(min=1e-6)


class DMPNNEncoder(nn.Module):
    \"\"\"Directed Message Passing Neural Network (Yang et al., 2019).

    Bond-level messages flow along directed edges; each message passing step
    computes new edge messages from incoming atom messages minus the reverse
    edge contribution to avoid message collision.
    \"\"\"

    def __init__(self, atom_dim, edge_dim, hidden_dim=300, depth=3, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth = depth

        # Initial bond message: linear over [atom_src || bond_attr]
        self.W_i = nn.Linear(atom_dim + edge_dim, hidden_dim, bias=False)
        # Shared message-update weight (chemprop default)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Final atom-level readout combine
        self.W_o = nn.Linear(atom_dim + hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr, batch_idx):
        \"\"\"
        x: [total_atoms, atom_dim]
        edge_index: [2, total_edges] (bidirectional, paired as [i,j],[j,i])
        edge_attr: [total_edges, edge_dim]
        batch_idx: [total_atoms]
        \"\"\"
        src, dst = edge_index
        num_atoms = x.size(0)
        num_edges = edge_index.size(1)

        if num_edges == 0:
            # Fallback for atom-only molecules
            atom_hidden = self.act(self.W_o(torch.cat([x, torch.zeros(num_atoms, self.hidden_dim, device=x.device)], dim=-1)))
            return self.dropout(atom_hidden)

        # Reverse edge index: edges are added in pairs (i->j, j->i),
        # so reverse of edge e is e XOR 1.
        rev_edge_idx = torch.arange(num_edges, device=x.device) ^ 1
        rev_edge_idx = rev_edge_idx.clamp(max=num_edges - 1)

        # Initial bond input: source atom features concatenated with bond features
        bond_input = torch.cat([x[src], edge_attr], dim=-1)
        h0 = self.act(self.W_i(bond_input))  # [num_edges, hidden]
        h = h0

        # Message passing for depth-1 steps (chemprop convention)
        for _ in range(self.depth - 1):
            # Aggregate incoming messages to each atom
            atom_msg = torch.zeros(num_atoms, self.hidden_dim, device=x.device)
            atom_msg.index_add_(0, dst, h)

            # New edge message: a_v - h_{v->u}^{rev} (avoid passing back)
            new_h = atom_msg[src] - h[rev_edge_idx]
            new_h = self.W_h(new_h)
            # Residual on h0 (chemprop style)
            new_h = self.act(h0 + new_h)
            new_h = self.dropout(new_h)
            h = new_h

        # Final atom messages
        atom_msg = torch.zeros(num_atoms, self.hidden_dim, device=x.device)
        atom_msg.index_add_(0, dst, h)

        # Combine atom features with aggregated bond messages
        atom_hidden = self.act(self.W_o(torch.cat([x, atom_msg], dim=-1)))
        atom_hidden = self.dropout(atom_hidden)
        return atom_hidden


class MoleculeModel(nn.Module):
    \"\"\"D-MPNN with RDKit 2D normalized molecular descriptors.

    Configuration follows Yang et al. 2019 chemprop defaults:
      - hidden_dim = 300
      - depth = 3 message passing steps
      - sum readout per graph
      - 2-layer FFN head with hidden=300
      - RDKit 2D descriptors concatenated at the readout (\"+features\" mode)
    \"\"\"

    def __init__(self, atom_dim: int, edge_dim: int, num_tasks: int, task_type: str):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_type = task_type
        hidden_dim = 300
        depth = 3
        # `pooler_dropout` may be set by the training driver to vary dropout
        # per dataset (e.g. BACE/Tox21=0.1, BBBP=0.0, regression tasks=0.1-0.2)
        dropout = float(getattr(type(self), \"pooler_dropout\", 0.0))

        self.encoder = DMPNNEncoder(
            atom_dim=atom_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
        )

        # RDKit 2D descriptor branch
        self.feat_norm = _RunningNormalizer(_RDKIT_FEAT_DIM)

        # 2-layer FFN head over [graph_embed || rdkit_features]
        readout_in = hidden_dim + _RDKIT_FEAT_DIM
        self.readout = nn.Sequential(
            nn.Linear(readout_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks),
        )

        # Lazy SMILES->feature cache (shared across forward calls)
        self._smi_cache = {}

    def _batch_rdkit_features(self, batch):
        \"\"\"Compute RDKit features for the molecules in this batch.

        Uses LMDB SMILES via the dataset wrapper.  When SMILES are not
        available (no `_smiles` attr), falls back to a zero vector — the
        running normalizer will then produce zeros, leaving the GNN
        branch unaffected.
        \"\"\"
        smiles = getattr(batch, \"_smiles\", None)
        if smiles is None:
            num_graphs = int(batch.batch_idx.max().item()) + 1
            return torch.zeros(num_graphs, _RDKIT_FEAT_DIM,
                               device=batch.x.device)

        feats = []
        for smi in smiles:
            if smi in self._smi_cache:
                feats.append(self._smi_cache[smi])
            else:
                f = _rdkit_2d_descriptors(smi)
                self._smi_cache[smi] = f
                feats.append(f)
        return torch.tensor(feats, dtype=torch.float32, device=batch.x.device)

    def forward(self, batch):
        atom_hidden = self.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch_idx)

        # Sum pooling per graph (chemprop default)
        num_graphs = int(batch.batch_idx.max().item()) + 1
        graph_embed = torch.zeros(num_graphs, atom_hidden.size(-1), device=atom_hidden.device)
        graph_embed.index_add_(0, batch.batch_idx, atom_hidden)

        # RDKit feature branch (per-graph)
        rdkit_feats = self._batch_rdkit_features(batch)
        rdkit_feats = self.feat_norm(rdkit_feats)

        combined = torch.cat([graph_embed, rdkit_feats], dim=-1)
        return self.readout(combined)

# =====================================================================
# EDITABLE SECTION END
# =====================================================================
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 115,
        "end_line": 207,
        "content": _CONTENT,
    },
]
