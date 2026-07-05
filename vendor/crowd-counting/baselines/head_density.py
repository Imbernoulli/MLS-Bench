"""Good baseline for cv-count-formulation: density-map integral.

Predict a NON-NEGATIVE per-pixel density map and count by its spatial integral (the
MCNN / CSRNet formulation): ~h*w densely-supervised spatially-local votes are summed,
giving a low-variance per-image count -> lower counting MAE with clear headroom over
direct scalar regression.
"""


def build_count_head(cin):
    import torch.nn as nn
    import torch.nn.functional as F
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, 64, 3, padding=1), nn.ReLU(True),
                nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(True),
                nn.Conv2d(32, 1, 1))
        def forward(self, feat):
            return F.softplus(self.net(feat)).squeeze(1)   # (B,h,w) density
    return Head()
