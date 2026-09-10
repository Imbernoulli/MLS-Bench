"""Barlow Twins baseline -- rigorous codebase edit ops.

Faithful port of Barlow Twins (Zbontar et al. ICML 2021). Unlike KoLeo
(DINOv2's minor auxiliary), Barlow Twins is specifically designed as a
standalone anti-collapse SSL objective and does not require a teacher/
student or EMA split — the same regime as our template.

Reference implementation:
https://github.com/facebookresearch/barlowtwins/blob/main/main.py

Loss (verbatim from the paper's forward):
    c = bn(z1).T @ bn(z2) / batch_size    # cross-correlation matrix
    on_diag  = (diagonal(c) - 1).pow(2).sum()
    off_diag = off_diagonal(c).pow(2).sum()
    loss = on_diag + lambd * off_diag

Paper defaults:
    lambd = 0.0051
    bn: nn.BatchNorm1d(D, affine=False)  — feature-wise standardization
        across the batch, no learnable affine.
    projector: 8192-8192-8192 for ImageNet ResNet-50; the common CIFAR-10
        ResNet-18 setting uses 2048-2048-2048, which is the template's
        default, so no CONFIG_OVERRIDES are needed.

Because the template instantiates CustomRegularizer() with no args and
we only see the feature dimension at first forward, the BatchNorm1d is
created lazily on the first call. Everything else — the cross-
correlation division by batch_size, the off_diagonal helper, the
lambda default — is verbatim.

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "eb_jepa/custom_regularizer.py"

# ── Replace editable region: CustomRegularizer class + CONFIG_OVERRIDES ─────

_BARLOW_CLASS = """\
class CustomRegularizer(nn.Module):
    \"\"\"Barlow Twins (Zbontar et al. ICML 2021).

    NB on scale_loss: the paper's official 8192-projector recipe includes
    a `--scale-loss 0.024` multiplier — see the README of the original
    repo's mirror (xuChenSJTU/barlowtwins-1) and solo-learn's reference
    implementation
        https://github.com/vturrisi/solo-learn/blob/main/solo/losses/barlow.py
    Without it the raw loss is on the order of 1e3-1e4, and LARS' adaptive
    rescaling (lars_lr = p_norm / (g_norm + ...)) starves the optimizer
    so the diagonal of the cross-correlation matrix never approaches 1.
    Using paper-default scale_loss=0.024 with the 8192 projector.
    \"\"\"

    def __init__(self, lambd=0.0051, scale_loss=0.1):
        super().__init__()
        self.lambd = lambd
        self.scale_loss = scale_loss
        # Use LazyBatchNorm1d so the module is registered in __init__
        # (with proper to(device)/dtype propagation) but the feature dim
        # is materialized on the first forward call.
        self.bn = nn.LazyBatchNorm1d(affine=False)

    @staticmethod
    def _off_diagonal(x):
        # Verbatim from barlowtwins/main.py.
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        B = z1.shape[0]

        # Cross-correlation matrix (paper forward, verbatim).
        c = self.bn(z1).T @ self.bn(z2)
        c = c / B

        on_diag = (torch.diagonal(c) - 1).pow(2).sum()
        off_diag = self._off_diagonal(c).pow(2).sum()
        total_loss = self.scale_loss * (on_diag + self.lambd * off_diag)

        return {
            "loss": total_loss,
            "on_diag": on_diag,
            "off_diag": off_diag,
        }


# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: proj_output_dim, proj_hidden_dim.
# Use the solo-learn CIFAR-10 Barlow Twins recipe (proj=2048,
# scale_loss=0.1) instead of the paper's ImageNet recipe
# (proj=8192, scale_loss=0.024, batch=2048, epochs=1000). Our setup
# matches solo-learn's: CIFAR-10, batch=256, ResNet-{18,34,50}, LARS
# with eta=0.02 and clip_lr=True. The paper's 8192 recipe needs
# epochs=1000 + batch=2048 to converge — at our 100-epoch budget it
# leaves the diagonal stuck (see logs from v3: rn34 only reaches 10%).
# https://github.com/vturrisi/solo-learn/blob/main/scripts/pretrain/cifar/barlow.yaml
CONFIG_OVERRIDES = {}
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 33,
        "end_line": 58,
        "content": _BARLOW_CLASS,
    },
]
