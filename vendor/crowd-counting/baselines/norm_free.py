"""Good baseline for cv-count-normalization: free non-negative density field.

A convolutional head with a per-pixel softplus: the total integrated mass is
UNBOUNDED, so the count can scale to match arbitrarily crowded scenes -> it
extrapolates to the higher-count val images -> lower counting MAE with clear headroom
over the softmax-normalised (bottlenecked-mass) head.
"""


def build_density_head(cin):
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
            return F.softplus(self.net(feat)).squeeze(1)
    return Head()
