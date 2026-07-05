"""Weak baseline (negative control) for cv-count-formulation: direct scalar count.

Global-average-pool -> MLP -> ONE scalar count. Discards spatial density structure ->
high-variance, regresses to the mean -> high counting MAE. This is the starting
default in vendor/crowd-counting/solution/head.py.
"""


def build_count_head(cin):
    import torch.nn as nn
    import torch.nn.functional as F
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(cin, 128), nn.ReLU(True),
                nn.Linear(128, 64), nn.ReLU(True),
                nn.Linear(64, 1))
        def forward(self, feat):
            pooled = feat.mean(dim=(-2, -1))
            return F.softplus(self.mlp(pooled)).squeeze(-1)
    return Head()
