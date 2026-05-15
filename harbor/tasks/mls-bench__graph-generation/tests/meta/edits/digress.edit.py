"""DiGress baseline for graph-generation.

Discrete denoising diffusion for graph generation. Uses a discrete noise
process that corrupts adjacency entries by flipping edges, and a graph
transformer to predict the clean graph at each denoising step.

Reference: Vignac et al., "DiGress: Discrete Denoising diffusion for
graph generation" (ICLR 2023)
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

class GraphTransformerLayer(nn.Module):
    \"\"\"Graph transformer layer with edge-aware attention.\"\"\"

    def __init__(self, dim, n_heads=4, ff_dim=None):
        super().__init__()
        ff_dim = ff_dim or 4 * dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.edge_bias = nn.Linear(1, n_heads)
        self.proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim),
        )

    def forward(self, x, adj):
        B, N, C = x.shape
        # Multi-head attention with edge bias
        q = self.q(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Edge bias
        edge_b = self.edge_bias(adj.unsqueeze(-1))  # [B, N, N, n_heads]
        attn = attn + edge_b.permute(0, 3, 1, 2)

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.norm1(x + self.proj(out))
        x = self.norm2(x + self.ff(x))
        return x


class DiscreteDenoiser(nn.Module):
    \"\"\"Denoiser network for discrete adjacency diffusion.\"\"\"

    def __init__(self, max_nodes, hidden_dim=128, n_layers=4, n_heads=4):
        super().__init__()
        self.node_embed = nn.Linear(max_nodes, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, n_heads)
            for _ in range(n_layers)
        ])
        self.edge_pred = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.node_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, adj_noisy, t):
        B, N, _ = adj_noisy.shape
        device = adj_noisy.device

        # Node features
        x = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
        x = self.node_embed(x)

        # Add time conditioning
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_embed(t).unsqueeze(1)  # [B, 1, hidden]
        x = x + t_emb

        # Graph transformer layers
        for layer in self.layers:
            x = layer(x, adj_noisy)

        # Edge prediction
        ni = x.unsqueeze(2).expand(-1, -1, N, -1)
        nj = x.unsqueeze(1).expand(-1, N, -1, -1)
        edge_input = torch.cat([ni, nj, adj_noisy.unsqueeze(-1)], dim=-1)
        edge_logits = self.edge_pred(edge_input).squeeze(-1)
        edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2
        mask = 1 - torch.eye(N, device=device).unsqueeze(0)
        edge_logits = edge_logits * mask

        # Node prediction
        node_logits = self.node_pred(x).squeeze(-1)

        return edge_logits, node_logits


class GraphGenerator(nn.Module):
    \"\"\"DiGress: Discrete denoising diffusion for graphs.

    Uses a discrete corruption process (edge flipping) and a graph
    transformer denoiser to predict the clean graph.

    Reference: Vignac et al., ICLR 2023.
    \"\"\"

    def __init__(self, max_nodes, hidden_dim=128, n_layers=4, n_heads=4,
                 n_diffusion_steps=50, lr=2e-4, **kwargs):
        super().__init__()
        self.max_nodes = max_nodes
        self.n_steps = n_diffusion_steps

        # Beta schedule: cosine schedule for discrete diffusion
        steps = torch.arange(n_diffusion_steps + 1, dtype=torch.float64)
        alpha_bar = torch.cos((steps / n_diffusion_steps + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        betas = torch.clamp(betas, max=0.999)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alpha_bar", alpha_bar[1:].float())

        self.denoiser = DiscreteDenoiser(max_nodes, hidden_dim, n_layers, n_heads)
        self.optimizer = optim.Adam(self.denoiser.parameters(), lr=lr)

    def _corrupt(self, adj, t_idx):
        \"\"\"Discrete corruption: flip edges with probability depending on t.\"\"\"
        B = adj.shape[0]
        device = adj.device

        # Flip probability = 0.5 * (1 - alpha_bar_t)
        alpha_bar_t = self.alpha_bar[t_idx].view(B, 1, 1)
        flip_prob = 0.5 * (1 - alpha_bar_t)

        # Sample flip mask
        flip_mask = (torch.rand_like(adj) < flip_prob).float()
        # Make symmetric
        flip_mask = torch.triu(flip_mask, diagonal=1)
        flip_mask = flip_mask + flip_mask.transpose(1, 2)

        # Apply flips: XOR with flip mask
        adj_noisy = torch.abs(adj - flip_mask)
        return adj_noisy

    def train_step(self, adj, node_counts):
        self.train()
        self.optimizer.zero_grad()
        B = adj.shape[0]
        device = adj.device

        # Sample random timestep
        t_idx = torch.randint(0, self.n_steps, (B,), device=device)

        # Corrupt adjacency
        adj_noisy = self._corrupt(adj, t_idx)

        # Predict clean adjacency
        t_float = t_idx.float() / self.n_steps
        edge_logits, node_logits = self.denoiser(adj_noisy, t_float)

        # Cross-entropy loss to predict original clean graph
        edge_loss = F.binary_cross_entropy_with_logits(edge_logits, adj, reduction="mean")

        # Node existence loss
        node_target = (adj.sum(dim=-1) > 0).float()
        node_loss = F.binary_cross_entropy_with_logits(node_logits, node_target, reduction="mean")

        loss = edge_loss + 0.5 * node_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
        self.optimizer.step()

        return {"loss": loss.item(), "edge_loss": edge_loss.item()}

    def sample(self, n_samples, device):
        \"\"\"Generate graphs via iterative discrete denoising.\"\"\"
        self.eval()
        N = self.max_nodes
        mask = 1 - torch.eye(N, device=device).unsqueeze(0)

        with torch.no_grad():
            # Start from random binary adjacency (Bernoulli(0.5))
            adj = (torch.rand(n_samples, N, N, device=device) > 0.5).float()
            adj = torch.triu(adj, diagonal=1)
            adj = adj + adj.transpose(1, 2)

            for step in range(self.n_steps - 1, -1, -1):
                t_float = torch.ones(n_samples, device=device) * (step / self.n_steps)
                edge_logits, node_logits = self.denoiser(adj, t_float)
                edge_probs = torch.sigmoid(edge_logits)

                if step > 0:
                    # Sample with some noise
                    adj = (torch.rand_like(edge_probs) < edge_probs).float()
                else:
                    # Final step: use threshold
                    adj = (edge_probs > 0.5).float()

                # Ensure symmetry and no self-loops
                adj = torch.triu(adj, diagonal=1)
                adj = adj + adj.transpose(1, 2)

            # Node counts from predictor
            node_probs = torch.sigmoid(node_logits)
            node_mask_pred = (node_probs > 0.5).float()
            adj = adj * node_mask_pred.unsqueeze(-1) * node_mask_pred.unsqueeze(-2)
            node_counts = node_mask_pred.sum(dim=-1).long()
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
