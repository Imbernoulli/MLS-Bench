"""GraphVAE baseline for graph-generation.

Variational autoencoder for graphs with graph-level encoding and
probabilistic adjacency matrix decoding. Uses GCN encoder for
graph-aware latent representations.

Reference: Simonovsky & Komodakis, "GraphVAE: Towards Generation of
Small Graphs Using Variational Autoencoders" (arXiv:1802.03480 / ICANN 2018)
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

class GCNLayer(nn.Module):
    \"\"\"Simple GCN layer: X' = D^{-1/2} A_hat D^{-1/2} X W.\"\"\"

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # Add self-loops
        I = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0)
        A_hat = adj + I
        # Degree normalization
        D = A_hat.sum(dim=-1, keepdim=True).clamp(min=1)
        D_inv_sqrt = 1.0 / torch.sqrt(D)
        A_norm = A_hat * D_inv_sqrt * D_inv_sqrt.transpose(-1, -2)
        out = torch.bmm(A_norm, x)
        return self.linear(out)


class GraphGenerator(nn.Module):
    \"\"\"GraphVAE: Variational Autoencoder for graph generation.

    Uses GCN encoder to produce graph-level latent representation,
    and MLP decoder to produce adjacency matrix probabilities.

    Reference: Simonovsky & Komodakis, arXiv:1802.03480 / ICANN 2018.
    \"\"\"

    def __init__(self, max_nodes, hidden_dim=256, latent_dim=64, lr=1e-3, **kwargs):
        super().__init__()
        self.max_nodes = max_nodes
        self.latent_dim = latent_dim
        adj_size = max_nodes * max_nodes

        # GCN encoder
        self.gcn1 = GCNLayer(max_nodes, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # MLP decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, adj_size),
        )

        # Node existence predictor
        self.node_pred = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_nodes),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def encode(self, adj):
        B, N, _ = adj.shape
        # Use identity as node features (one-hot position)
        x = torch.eye(N, device=adj.device).unsqueeze(0).expand(B, -1, -1)
        h = F.relu(self.gcn1(x, adj))
        h = F.relu(self.gcn2(h, adj))
        # Graph-level readout (mean pooling)
        h_graph = h.mean(dim=1)  # [B, hidden]
        return self.fc_mu(h_graph), self.fc_logvar(h_graph)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        B = z.shape[0]
        logits = self.decoder(z).view(B, self.max_nodes, self.max_nodes)
        # Symmetrize
        logits = (logits + logits.transpose(1, 2)) / 2
        # Zero diagonal
        mask = 1 - torch.eye(self.max_nodes, device=z.device).unsqueeze(0)
        return logits * mask

    def train_step(self, adj, node_counts):
        self.train()
        self.optimizer.zero_grad()

        mu, logvar = self.encode(adj)
        z = self.reparameterize(mu, logvar)
        adj_logits = self.decode(z)
        node_logits = self.node_pred(z)  # [B, max_nodes]

        # Reconstruction loss
        recon_loss = F.binary_cross_entropy_with_logits(adj_logits, adj, reduction="mean")

        # Node existence loss
        node_target = (adj.sum(dim=-1) > 0).float()  # [B, max_nodes]
        node_loss = F.binary_cross_entropy_with_logits(node_logits, node_target, reduction="mean")

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recon_loss + 0.5 * node_loss + 0.001 * kl_loss
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "recon": recon_loss.item(), "kl": kl_loss.item()}

    def sample(self, n_samples, device):
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=device)
            adj_logits = self.decode(z)
            adj = (torch.sigmoid(adj_logits) > 0.5).float()
            node_logits = self.node_pred(z)
            node_probs = torch.sigmoid(node_logits)
            node_mask = (node_probs > 0.5).float()
            # Mask adjacency by existing nodes
            adj = adj * node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
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
