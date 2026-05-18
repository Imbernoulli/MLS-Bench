"""GDSS baseline for graph-generation.

Score-based generative model for graphs using a system of stochastic
differential equations (SDEs). Jointly models node and adjacency
diffusion processes with score matching.

Reference: Jo et al., "Score-based Generative Modeling of Graphs via
the System of Stochastic Differential Equations" (ICML 2022)
"""

_FILE = "pytorch-geometric/custom_graphgen.py"

_CONTENT = """\
# The agent should modify the GraphGenerator class below.
# The class must implement:
#   - __init__(self, max_nodes, **kwargs): initialize model parameters
#   - train_step(self, adj, node_counts) -> dict: one training step, returns loss dict
#   - sample(self, n_samples, device) -> (adj_matrices, node_counts):
#       generate n_samples graphs, return adjacency tensors and node count tensets
#
# The model receives adjacency matrices [B, max_nodes, max_nodes] and node counts [B].
# It should generate adjacency matrices of similar structure.
# ============================================================================

class ScoreNetwork(nn.Module):
    \"\"\"Score network for adjacency matrix diffusion.

    Predicts the score (gradient of log density) of the noisy adjacency
    distribution at a given noise level.
    \"\"\"

    def __init__(self, max_nodes, hidden_dim=256, n_layers=3):
        super().__init__()
        self.max_nodes = max_nodes
        adj_size = max_nodes * max_nodes

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # GCN-style layers for graph-aware processing
        layers = []
        in_dim = max_nodes + hidden_dim  # node features + time
        for i in range(n_layers):
            out_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.SiLU())
            in_dim = out_dim
        self.node_net = nn.Sequential(*layers)

        # Edge score prediction from node pairs
        self.edge_score = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, adj_noisy, t):
        \"\"\"Predict score of adjacency matrix.

        Args:
            adj_noisy: [B, N, N] noisy adjacency
            t: [B] or [B, 1] diffusion time

        Returns:
            score: [B, N, N] predicted score
        \"\"\"
        B, N, _ = adj_noisy.shape
        device = adj_noisy.device

        # Time embedding
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_embed(t)  # [B, hidden]
        t_emb = t_emb.unsqueeze(1).expand(-1, N, -1)  # [B, N, hidden]

        # Node features from adjacency + GCN aggregation
        I = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
        A_hat = adj_noisy + I
        D = A_hat.sum(dim=-1, keepdim=True).clamp(min=1)
        A_norm = A_hat / D
        x = torch.bmm(A_norm, torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1))

        # Concatenate with time
        x = torch.cat([x, t_emb], dim=-1)  # [B, N, N+hidden]
        node_feat = self.node_net(x)  # [B, N, hidden]

        # Predict edge scores from node pairs
        ni = node_feat.unsqueeze(2).expand(-1, -1, N, -1)
        nj = node_feat.unsqueeze(1).expand(-1, N, -1, -1)
        edge_input = torch.cat([ni, nj, adj_noisy.unsqueeze(-1)], dim=-1)
        score = self.edge_score(edge_input).squeeze(-1)  # [B, N, N]

        # Symmetrize and zero diagonal
        score = (score + score.transpose(1, 2)) / 2
        mask = 1 - torch.eye(N, device=device).unsqueeze(0)
        score = score * mask

        return score


class GraphGenerator(nn.Module):
    \"\"\"GDSS: Score-based graph generation via SDEs.

    Uses VP-SDE (Variance Preserving) for the adjacency diffusion process
    and trains a score network via denoising score matching.

    Reference: Jo et al., ICML 2022.
    \"\"\"

    def __init__(self, max_nodes, hidden_dim=256, n_layers=3, beta_min=0.1,
                 beta_max=1.0, n_diffusion_steps=100, lr=2e-4, **kwargs):
        super().__init__()
        self.max_nodes = max_nodes
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.n_steps = n_diffusion_steps

        self.score_net = ScoreNetwork(max_nodes, hidden_dim, n_layers)
        self.optimizer = optim.Adam(self.score_net.parameters(), lr=lr)

    def _beta(self, t):
        \"\"\"Linear beta schedule.\"\"\"
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _marginal_params(self, t):
        \"\"\"Compute mean coefficient and std for VP-SDE marginal q(x_t|x_0).\"\"\"
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean_coeff = torch.exp(log_mean_coeff)
        std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff))
        return mean_coeff, std

    def train_step(self, adj, node_counts):
        self.train()
        self.optimizer.zero_grad()
        B = adj.shape[0]
        device = adj.device

        # Sample random time
        t = torch.rand(B, device=device) * 0.998 + 0.001  # [0.001, 0.999]

        # Forward diffusion: add noise
        mean_coeff, std = self._marginal_params(t)
        mean_coeff = mean_coeff.view(B, 1, 1)
        std = std.view(B, 1, 1)

        noise = torch.randn_like(adj)
        noise = (noise + noise.transpose(1, 2)) / math.sqrt(2)  # Symmetric noise
        mask = 1 - torch.eye(self.max_nodes, device=device).unsqueeze(0)
        noise = noise * mask

        adj_noisy = mean_coeff * adj + std * noise

        # Predict score (which equals -noise/std for VP-SDE)
        score_pred = self.score_net(adj_noisy, t)

        # Score matching loss: ||score_pred + noise/std||^2
        target = -noise / std.clamp(min=1e-5)
        loss = F.mse_loss(score_pred * mask, target * mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.score_net.parameters(), 1.0)
        self.optimizer.step()

        return {"loss": loss.item()}

    def sample(self, n_samples, device):
        \"\"\"Generate graphs via reverse SDE (Euler-Maruyama).\"\"\"
        self.eval()
        N = self.max_nodes
        mask = 1 - torch.eye(N, device=device).unsqueeze(0)

        with torch.no_grad():
            # Start from noise
            adj = torch.randn(n_samples, N, N, device=device)
            adj = (adj + adj.transpose(1, 2)) / math.sqrt(2)
            adj = adj * mask

            dt = 1.0 / self.n_steps
            for i in range(self.n_steps, 0, -1):
                t = torch.ones(n_samples, device=device) * (i * dt)
                beta_t = self._beta(t)
                _, std_t = self._marginal_params(t)

                score = self.score_net(adj, t)

                # Reverse SDE step
                drift = -0.5 * beta_t.view(-1, 1, 1) * (adj + score)
                diffusion = torch.sqrt(beta_t).view(-1, 1, 1)

                noise = torch.randn_like(adj)
                noise = (noise + noise.transpose(1, 2)) / math.sqrt(2)
                noise = noise * mask

                adj = adj - drift * dt + diffusion * math.sqrt(dt) * noise
                adj = adj * mask

            # Threshold to binary adjacency matrix
            adj = (adj > 0.0).float()
            # Ensure symmetry
            adj = torch.triu(adj, diagonal=1)
            adj = adj + adj.transpose(1, 2)

            # Node counts
            node_mask = (adj.sum(dim=-1) > 0).float()
            node_counts = node_mask.sum(dim=-1).long()
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
