"""VICReg baseline -- rigorous codebase edit ops.

Replaces the placeholder CustomRegularizer with a VICReg implementation
that combines invariance (MSE), variance (hinge std), and covariance
(off-diagonal decorrelation) losses.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "eb_jepa/custom_regularizer.py"

# ── Replace CustomRegularizer class (lines 33-53) ───────────────────────────

_VICREG_CLASS = """\
class CustomRegularizer(nn.Module):
    \"\"\"VICReg: Variance-Invariance-Covariance Regularization.\"\"\"

    def __init__(self, std_coeff=1.0, cov_coeff=100.0, std_margin=1.0):
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.std_margin = std_margin

    def _std_loss(self, x):
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0) + 0.0001)
        return torch.mean(F.relu(self.std_margin - std))

    def _off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _cov_loss(self, x):
        batch_size = x.shape[0]
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / (batch_size - 1)
        return self._off_diagonal(cov).pow(2).mean()

    def forward(self, z1, z2):
        sim_loss = F.mse_loss(z1, z2)
        var_loss = self._std_loss(z1) + self._std_loss(z2)
        cov_loss = self._cov_loss(z1) + self._cov_loss(z2)
        total_loss = sim_loss + self.std_coeff * var_loss + self.cov_coeff * cov_loss
        return {
            "loss": total_loss,
            "invariance_loss": sim_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
        }


# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: proj_output_dim, proj_hidden_dim.
# Paper README "Impact of the projector" table ranks VICReg's best
# projector as 2048->1024 (90.12% on CIFAR-10 ResNet-18, 300 epochs).
CONFIG_OVERRIDES = {"proj_output_dim": 1024}
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 33,
        "end_line": 58,
        "content": _VICREG_CLASS,
    },
]
