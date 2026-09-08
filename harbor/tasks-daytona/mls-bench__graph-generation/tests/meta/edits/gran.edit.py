"""GRAN baseline for graph-generation.

Graph Recurrent Attention Network that generates graphs by iteratively
adding blocks of edges using attention-based message passing. Uses a
one-shot approach with iterative refinement.

Reference: Liao et al., "Efficient Graph Generation with Graph Recurrent
Attention Networks" (NeurIPS 2019)
"""

_FILE = "pytorch-geometric/custom_graphgen.py"

_CONTENT = """\
# The agent should modify the GraphGenerator class below.
# The class must implement:
#   - __init__(self, max_nodes, **kwargs): initialize model parameters
#   - train_step(self, adj, node_counts) -> dict: one training step, returns loss dict
#   - sample(self, n_samples, device) -> (adj_matrices, node_counts):
#       generate n_samples graphs, return adjacency tensors and node count tensors
#
# The model receives adjacency matrices [B, max_nodes, max_nodes] and node counts [B].
# It should generate adjacency matrices of similar structure.
# ============================================================================

class AttentionBlock(nn.Module):
    \"\"\"Multi-head attention for graph nodes.\"\"\"

    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.norm(x + self.proj(out))


class GRANBlock(nn.Module):
    \"\"\"GRAN message passing block with attention and edge prediction.\"\"\"

    def __init__(self, node_dim, edge_dim=1, n_heads=4):
        super().__init__()
        self.attn = AttentionBlock(node_dim, n_heads)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, edge_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, node_feat, edge_feat, mask=None):
        B, N, D = node_feat.shape
        # Attention-based node update
        node_feat = self.attn(node_feat, mask)

        # Edge update
        ni = node_feat.unsqueeze(2).expand(-1, -1, N, -1)
        nj = node_feat.unsqueeze(1).expand(-1, N, -1, -1)
        edge_input = torch.cat([ni, nj, edge_feat], dim=-1)
        edge_feat = edge_feat + self.edge_mlp(edge_input)

        # Aggregate edge info to nodes
        if mask is not None:
            edge_agg = (edge_feat * mask.unsqueeze(-1).unsqueeze(-1).float()).sum(dim=2)
        else:
            edge_agg = edge_feat.mean(dim=2)
        node_input = torch.cat([node_feat, edge_agg], dim=-1)
        node_feat = self.norm(node_feat + self.node_mlp(node_input))

        return node_feat, edge_feat


class GraphGenerator(nn.Module):
    \"\"\"GRAN: Graph Recurrent Attention Network.

    Iteratively refines node and edge representations using attention-based
    message passing, then predicts edge probabilities for graph generation.

    Reference: Liao et al., NeurIPS 2019.
    \"\"\"

    def __init__(self, max_nodes, hidden_dim=128, n_layers=3, n_heads=4,
                 n_refine_steps=5, lr=1e-3, **kwargs):
        super().__init__()
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim
        self.n_refine_steps = n_refine_steps

        # Node embedding
        self.node_embed = nn.Linear(max_nodes, hidden_dim)

        # GRAN blocks (shared across refinement steps)
        self.blocks = nn.ModuleList([
            GRANBlock(hidden_dim, edge_dim=1, n_heads=n_heads)
            for _ in range(n_layers)
        ])

        # Final edge prediction
        self.edge_pred = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Node existence prediction
        self.node_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def _forward(self, adj, node_mask=None):
        B, N, _ = adj.shape
        device = adj.device

        # Initial node features (identity-based)
        x = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
        node_feat = F.relu(self.node_embed(x))  # [B, N, hidden]

        # Initial edge features from adjacency
        edge_feat = adj.unsqueeze(-1)  # [B, N, N, 1]

        # Iterative refinement
        for block in self.blocks:
            node_feat, edge_feat = block(node_feat, edge_feat, node_mask)

        # Predict edges
        ni = node_feat.unsqueeze(2).expand(-1, -1, N, -1)
        nj = node_feat.unsqueeze(1).expand(-1, N, -1, -1)
        edge_input = torch.cat([ni, nj, edge_feat], dim=-1)
        edge_logits = self.edge_pred(edge_input).squeeze(-1)  # [B, N, N]

        # Symmetrize and remove self-loops
        edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2
        diag_mask = 1 - torch.eye(N, device=device).unsqueeze(0)
        edge_logits = edge_logits * diag_mask

        # Node existence
        node_logits = self.node_pred(node_feat).squeeze(-1)  # [B, N]

        return edge_logits, node_logits

    def train_step(self, adj, node_counts):
        self.train()
        self.optimizer.zero_grad()

        edge_logits, node_logits = self._forward(adj)

        # Edge loss
        edge_loss = F.binary_cross_entropy_with_logits(edge_logits, adj, reduction="mean")

        # Node existence loss
        node_target = (adj.sum(dim=-1) > 0).float()
        node_loss = F.binary_cross_entropy_with_logits(node_logits, node_target, reduction="mean")

        loss = edge_loss + 0.5 * node_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        return {"loss": loss.item(), "edge_loss": edge_loss.item()}

    def sample(self, n_samples, device):
        self.eval()
        with torch.no_grad():
            # Start from random sparse adjacency (not zeros) to provide edge signal.
            # Starting from zeros leads to empty output: node aggregation is zero ->
            # node predictor predicts all nodes absent -> empty graph.
            p_init = 0.3
            adj = (torch.rand(n_samples, self.max_nodes, self.max_nodes, device=device) < p_init).float()
            adj = torch.triu(adj, diagonal=1)
            adj = adj + adj.transpose(1, 2)

            for step in range(self.n_refine_steps):
                edge_logits, node_logits = self._forward(adj)
                edge_probs = torch.sigmoid(edge_logits)
                adj = (torch.rand_like(edge_probs) < edge_probs).float()
                adj = torch.triu(adj, diagonal=1)
                adj = adj + adj.transpose(1, 2)

            # Derive node counts from connectivity (node predictor is unreliable
            # when initialized from random adjacency at inference time).
            node_counts = (adj.sum(dim=-1) > 0).long().sum(dim=-1)
            node_counts = torch.clamp(node_counts, min=2)

        return adj, node_counts

"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 446,
        "end_line": 590,
        "content": _CONTENT,
    },
]
