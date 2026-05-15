"""Baseline: L1 + LPIPS perceptual + KL divergence.

Standard practice for VAE training. Uses LPIPS (learned perceptual similarity)
for perceptual quality on top of L1 reconstruction.
"""

_FILE = "diffusers-main/custom_train.py"

_PERCEPTUAL = '''
class VAELoss(nn.Module):
    """Perceptual VAE loss: L1 + LPIPS + KL."""

    def __init__(self, device):
        super().__init__()
        import lpips
        self.lpips_fn = lpips.LPIPS(net='vgg').to(device)
        self.lpips_fn.eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad_(False)
        self.kl_weight = 1e-6
        self.perceptual_weight = 0.5

    def forward(self, recon, target, posterior, step):
        rec_loss = F.l1_loss(recon, target)
        p_loss = self.lpips_fn(recon.float(), target.float()).mean()
        kl_loss = posterior.kl().mean()
        loss = rec_loss + self.perceptual_weight * p_loss + self.kl_weight * kl_loss
        return loss, {
            "rec_loss": rec_loss.item(),
            "p_loss": p_loss.item(),
            "kl_loss": kl_loss.item(),
        }

'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 32,
        "end_line": 76,
        "content": _PERCEPTUAL,
    },
]
