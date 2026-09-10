"""AttentiveFP baseline — Attentive Fingerprint.
Replaces editable section with graph attention + GRU-based molecular fingerprint.
Reference: Xiong et al., "Pushing the Boundaries of Molecular Representation for Drug Discovery" (J. Med. Chem. 2020)
"""

_FILE = "Uni-Mol/custom_molprop.py"

_CONTENT = """\
# =====================================================================
# EDITABLE SECTION START — AttentiveFP: Attentive Fingerprint
# =====================================================================

def _scatter_softmax(logits, index, num_nodes):
    \"\"\"Numerically-stable scatter softmax with proper gradient flow.

    Uses only index_add_ (well-tested autograd) instead of index_reduce_
    with 'amax' which has broken/missing gradients in many PyTorch versions.

    Args:
        logits: [E, *] raw attention scores
        index: [E] destination indices for grouping
        num_nodes: number of groups
    Returns:
        softmax weights: [E, *] normalized per group
    \"\"\"
    # Detached max for numerical stability (no gradient needed through max)
    with torch.no_grad():
        max_vals = torch.full((num_nodes, *logits.shape[1:]),
                              float('-inf'), device=logits.device)
        max_vals.index_reduce_(0, index, logits.detach(), 'amax',
                               include_self=False)
        max_vals = max_vals.clamp(min=-1e4)
    exp_logits = torch.exp(logits - max_vals[index])
    sum_exp = torch.zeros(num_nodes, *logits.shape[1:], device=logits.device)
    sum_exp.index_add_(0, index, exp_logits)
    return exp_logits / sum_exp[index].clamp(min=1e-8)


class GATLayer(nn.Module):
    \"\"\"Graph Attention layer for AttentiveFP.\"\"\"

    def __init__(self, in_dim, out_dim, edge_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.out_dim = out_dim
        assert out_dim % num_heads == 0

        self.W_q = nn.Linear(in_dim, out_dim, bias=False)
        self.W_k = nn.Linear(in_dim, out_dim, bias=False)
        self.W_v = nn.Linear(in_dim, out_dim, bias=False)
        self.edge_proj = nn.Linear(edge_dim, num_heads)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.feat_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, edge_attr, batch_idx):
        \"\"\"
        x: [N, in_dim], edge_index: [2, E], edge_attr: [E, edge_dim]
        \"\"\"
        src, dst = edge_index
        N = x.size(0)

        q = self.W_q(x).view(N, self.num_heads, self.head_dim)
        k = self.W_k(x).view(N, self.num_heads, self.head_dim)
        v = self.W_v(x).view(N, self.num_heads, self.head_dim)

        # Attention scores for each edge
        q_dst = q[dst]  # [E, H, D]
        k_src = k[src]  # [E, H, D]
        attn = (q_dst * k_src).sum(-1) / math.sqrt(self.head_dim)  # [E, H]

        # Add edge bias
        edge_bias = self.edge_proj(edge_attr)  # [E, H]
        attn = attn + edge_bias

        # Softmax over incoming edges per destination node
        attn = _scatter_softmax(attn, dst, N)
        attn = self.dropout(attn)

        # Weighted message aggregation
        msg = v[src] * attn.unsqueeze(-1)  # [E, H, D]
        msg_flat = msg.view(-1, self.out_dim)  # [E, out_dim]
        agg = torch.zeros(N, self.out_dim, device=x.device)
        agg.index_add_(0, dst, msg_flat)

        out = self.out_proj(agg)
        # Residual connection + LayerNorm
        if x.size(-1) == out.size(-1):
            out = self.norm(out + x)
        else:
            out = self.norm(out)
        return self.feat_dropout(out)


class AttentiveReadout(nn.Module):
    \"\"\"GRU-based attentive readout for graph-level representation.\"\"\"

    def __init__(self, hidden_dim, num_steps=3):
        super().__init__()
        self.num_steps = num_steps
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_feats, batch_idx):
        num_graphs = batch_idx.max().item() + 1
        device = node_feats.device

        # Initialize graph embedding as mean of node features
        graph_embed = torch.zeros(num_graphs, node_feats.size(-1), device=device)
        counts = torch.zeros(num_graphs, 1, device=device)
        graph_embed.index_add_(0, batch_idx, node_feats)
        counts.index_add_(0, batch_idx, torch.ones(node_feats.size(0), 1, device=device))
        graph_embed = graph_embed / counts.clamp(min=1)

        for _ in range(self.num_steps):
            # Attention over nodes using current graph embedding
            expanded = graph_embed[batch_idx]  # [N, D]
            attn_input = torch.cat([node_feats, expanded], dim=-1)
            attn_weights = self.attn(attn_input)  # [N, 1]

            # Softmax per graph (using robust scatter softmax)
            attn_weights = _scatter_softmax(attn_weights, batch_idx, num_graphs)

            # Weighted sum
            context = torch.zeros(num_graphs, node_feats.size(-1), device=device)
            context.index_add_(0, batch_idx, node_feats * attn_weights)

            # GRU update
            graph_embed = self.gru(context, graph_embed)

        return graph_embed


class MoleculeModel(nn.Module):
    \"\"\"AttentiveFP: Graph Attention + GRU readout for molecular fingerprints.

    Uses multi-head graph attention layers to build atom representations,
    then an attentive GRU readout to learn a molecule-level fingerprint.
    \"\"\"

    def __init__(self, atom_dim: int, edge_dim: int, num_tasks: int, task_type: str):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_type = task_type
        hidden_dim = 256
        num_layers = 3

        self.atom_embed = nn.Linear(atom_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GATLayer(hidden_dim, hidden_dim, edge_dim, num_heads=4, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.readout = AttentiveReadout(hidden_dim, num_steps=3)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_tasks),
        )

    def forward(self, batch):
        x = self.atom_embed(batch.x)

        for layer in self.layers:
            x = layer(x, batch.edge_index, batch.edge_attr, batch.batch_idx)

        graph_embed = self.readout(x, batch.batch_idx)
        return self.head(graph_embed)

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
