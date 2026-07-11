"""SOTA pooling: Generalised-Mean (GeM) pooling with a learnable exponent p.
GeM interpolates between average (p=1) and max (p->inf) pooling and is the
strongest for image retrieval (Radenovic et al., 2018). Reference:
vendor/torchreid-reid/baselines/pool_gem.py

Standalone reference implementation of this baseline (also applied as an edit).
"""


def build_pooling():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class GeM(nn.Module):
        def __init__(self, p=3.0, eps=1e-6):
            super().__init__()
            self.p = nn.Parameter(torch.ones(1) * p)
            self.eps = eps
            self.name = "gem"

        def forward(self, x):
            x = x.clamp(min=self.eps).pow(self.p)
            x = F.adaptive_avg_pool2d(x, 1)
            return x.pow(1.0 / self.p)

    return GeM()
