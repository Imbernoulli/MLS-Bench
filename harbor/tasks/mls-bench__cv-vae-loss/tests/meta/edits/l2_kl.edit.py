"""Baseline: L1 reconstruction + KL divergence.

The simplest VAE loss: L1 pixel error (sharper than L2) and KL divergence
for latent space regularization.
"""

_FILE = "diffusers-main/custom_train.py"

_L2_KL = '''
class VAELoss(nn.Module):
    """Basic VAE loss: L1 reconstruction + KL divergence."""

    def __init__(self, device):
        super().__init__()
        self.kl_weight = 1e-6

    def forward(self, recon, target, posterior, step):
        rec_loss = F.l1_loss(recon, target)
        kl_loss = posterior.kl().mean()
        loss = rec_loss + self.kl_weight * kl_loss
        return loss, {
            "rec_loss": rec_loss.item(),
            "kl_loss": kl_loss.item(),
        }

'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 32,
        "end_line": 76,
        "content": _L2_KL,
    },
]
