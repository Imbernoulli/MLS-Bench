"""Baseline: L1 + LPIPS + KL + NLayerDiscriminator.

Implements the L1 + LPIPS + KL + PatchGAN loss family used by taming/VQGAN-
style autoencoder training, adapted to this benchmark's template loop.
The discriminator is trained via alternating steps in the template loop.
"""

_FILE = "diffusers-main/custom_train.py"

_FREQ = '''
class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator from taming-transformers/VQGAN."""
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(input_nc, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1),
        ]
        self.net = nn.Sequential(*layers)
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
    def forward(self, x):
        return self.net(x)


class VAELoss(nn.Module):
    """L1 + LPIPS + KL + PatchGAN loss for this benchmark harness."""
    def __init__(self, device):
        super().__init__()
        import lpips
        self.lpips_fn = lpips.LPIPS(net='vgg').to(device)
        self.lpips_fn.eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad_(False)
        self.disc = NLayerDiscriminator().to(device)
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.disc_start = 5000
        self.kl_weight = 1e-6
        self.perceptual_weight = 0.5

    def forward(self, recon, target, posterior, step):
        rf, tf = recon.float(), target.float()
        rec_loss = F.l1_loss(rf, tf)
        p_loss = self.lpips_fn(rf, tf).mean()
        kl_loss = posterior.kl().mean()
        loss = rec_loss + self.perceptual_weight * p_loss + self.kl_weight * kl_loss
        # Store perceptual loss tensor for adaptive weight computation in training loop
        self._perceptual_loss = self.perceptual_weight * p_loss
        return loss, {"rec": rec_loss.item(), "lpips": p_loss.item(),
                       "kl": kl_loss.item()}

'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 32,
        "end_line": 76,
        "content": _FREQ,
    },
]
