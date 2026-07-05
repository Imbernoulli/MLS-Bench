"""Weak baseline (negative control) for cv-count-normalization: softmax x scalar.

A spatial softmax (mass=1, pure location distribution) times a single learned count
scalar. The scalar saturates at the low training mean and cannot scale up to the
higher-count val images -> high counting MAE. This is the starting default in
vendor/crowd-counting/solution/norm.py.
"""


def build_density_head(cin):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, 64, 3, padding=1), nn.ReLU(True),
                nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(True),
                nn.Conv2d(32, 1, 1))
            self.count_scalar = nn.Parameter(torch.tensor(50.0))
        def forward(self, feat):
            m = self.net(feat)
            B, _, h, w = m.shape
            dist = F.softmax(m.view(B, -1), dim=1).view(B, h, w)
            total = F.softplus(self.count_scalar) * 100.0
            return dist * total
    return Head()
