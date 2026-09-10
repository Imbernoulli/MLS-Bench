"""MoFlow baseline for graph-generation.

Normalizing flow model for graph generation using invertible
transformations on adjacency matrices. Uses Glow-style coupling layers
adapted for graph structure.

Reference: Zang & Wang, "MoFlow: An Invertible Flow Model for
Generating Molecular Graphs" (KDD 2020)
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

class AffineCoupling(nn.Module):
    \"\"\"Affine coupling layer for flow-based model.

    Splits input, uses one half to predict scale and translation for the other.
    \"\"\"

    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        half_dim = dim // 2
        self.net = nn.Sequential(
            nn.Linear(half_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * (dim - half_dim)),
        )
        self.half_dim = half_dim

    def forward(self, x):
        \"\"\"Forward: data -> latent. Returns (z, log_det).\"\"\"
        x1, x2 = x[:, :self.half_dim], x[:, self.half_dim:]
        params = self.net(x1)
        s, t = params.chunk(2, dim=-1)
        s = torch.tanh(s) * 2  # Bounded scale
        z2 = x2 * torch.exp(s) + t
        log_det = s.sum(dim=-1)
        return torch.cat([x1, z2], dim=-1), log_det

    def inverse(self, z):
        \"\"\"Inverse: latent -> data.\"\"\"
        z1, z2 = z[:, :self.half_dim], z[:, self.half_dim:]
        params = self.net(z1)
        s, t = params.chunk(2, dim=-1)
        s = torch.tanh(s) * 2
        x2 = (z2 - t) * torch.exp(-s)
        return torch.cat([z1, x2], dim=-1)


class ActNorm(nn.Module):
    \"\"\"Activation normalization layer (data-dependent init).\"\"\"

    def __init__(self, dim):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.initialized = False

    def forward(self, x):
        if not self.initialized and self.training:
            with torch.no_grad():
                self.bias.data = -x.mean(dim=0)
                self.log_scale.data = -torch.log(x.std(dim=0) + 1e-6)
                self.initialized = True
        z = (x + self.bias) * torch.exp(self.log_scale)
        log_det = self.log_scale.sum().expand(x.shape[0])
        return z, log_det

    def inverse(self, z):
        return z * torch.exp(-self.log_scale) - self.bias


class FlowBlock(nn.Module):
    \"\"\"One block of the flow: ActNorm + AffineCoupling.\"\"\"

    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.actnorm = ActNorm(dim)
        self.coupling = AffineCoupling(dim, hidden_dim)

    def forward(self, x):
        z, log_det1 = self.actnorm(x)
        z, log_det2 = self.coupling(z)
        return z, log_det1 + log_det2

    def inverse(self, z):
        x = self.coupling.inverse(z)
        x = self.actnorm.inverse(x)
        return x


class GraphGenerator(nn.Module):
    \"\"\"MoFlow: Normalizing flow for graph generation.

    Uses a sequence of invertible transformations (ActNorm + affine coupling)
    on flattened upper-triangular adjacency matrices. Trained by maximizing
    the log-likelihood via change of variables.

    Reference: Zang & Wang, KDD 2020.
    \"\"\"

    def __init__(self, max_nodes, hidden_dim=256, n_flow_layers=6,
                 lr=1e-3, **kwargs):
        super().__init__()
        self.max_nodes = max_nodes
        # Work with upper triangular entries only
        self.tri_size = max_nodes * (max_nodes - 1) // 2

        # Flow blocks
        self.flows = nn.ModuleList([
            FlowBlock(self.tri_size, hidden_dim)
            for _ in range(n_flow_layers)
        ])

        # Node count predictor (separate from flow)
        self.node_pred = nn.Sequential(
            nn.Linear(self.tri_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_nodes),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def _adj_to_tri(self, adj):
        \"\"\"Extract upper triangular entries from adjacency matrices.\"\"\"
        B, N, _ = adj.shape
        idx = torch.triu_indices(N, N, offset=1)
        return adj[:, idx[0], idx[1]]  # [B, tri_size]

    def _tri_to_adj(self, tri):
        \"\"\"Reconstruct adjacency from upper triangular entries.\"\"\"
        B = tri.shape[0]
        N = self.max_nodes
        adj = torch.zeros(B, N, N, device=tri.device)
        idx = torch.triu_indices(N, N, offset=1)
        adj[:, idx[0], idx[1]] = tri
        adj = adj + adj.transpose(1, 2)
        return adj

    def forward_flow(self, x):
        \"\"\"Forward pass through all flow blocks.\"\"\"
        total_log_det = 0
        z = x
        for flow in self.flows:
            z, log_det = flow(z)
            total_log_det += log_det
        return z, total_log_det

    def inverse_flow(self, z):
        \"\"\"Inverse pass through all flow blocks.\"\"\"
        x = z
        for flow in reversed(self.flows):
            x = flow.inverse(x)
        return x

    def train_step(self, adj, node_counts):
        self.train()
        self.optimizer.zero_grad()

        # Convert to upper triangular + add small noise for continuous flow
        tri = self._adj_to_tri(adj)  # [B, tri_size]
        tri_noisy = tri + torch.randn_like(tri) * 0.05  # Dequantization noise

        # Forward flow
        z, log_det = self.forward_flow(tri_noisy)

        # Log-likelihood under standard normal prior
        prior_ll = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=-1)
        nll = -(prior_ll + log_det).mean()

        # Node prediction loss
        node_target = (adj.sum(dim=-1) > 0).float()
        node_logits = self.node_pred(tri)
        node_loss = F.binary_cross_entropy_with_logits(node_logits, node_target, reduction="mean")

        loss = nll + 0.5 * node_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        return {"loss": loss.item(), "nll": nll.item()}

    def sample(self, n_samples, device):
        \"\"\"Generate graphs by sampling from latent space and inverting flow.\"\"\"
        self.eval()
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(n_samples, self.tri_size, device=device) * 0.6  # Temperature

            # Inverse flow
            tri = self.inverse_flow(z)

            # Threshold to binary
            adj = self._tri_to_adj((tri > 0.0).float())

            # Node counts from adjacency
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
