"""mono3d-head-capacity STRONG baseline: DEEP/WIDE residual refinement block.

Refine the shared embedding with a wider, deeper residual MLP (emb -> 2*emb -> emb with a skip
connection, x2 blocks). The residual/skip preserves the encoded information while adding capacity
to disentangle depth/pose factors before the task heads, and the width avoids any bottleneck ->
higher AP3D. This is the well-capacity reference. Reference: standard capacity ablation — a
sufficiently wide residual refinement improves the representation without vanishing gradients.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_backbone(emb_dim):
    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(),
                                   nn.Linear(2 * emb_dim, emb_dim))

        def forward(self, x):
            return F.relu(x + self.f(x))     # residual: preserves info + adds capacity

    class _Deep(nn.Module):
        def __init__(self):
            super().__init__()
            self.b1 = _Block()
            self.b2 = _Block()

        def forward(self, x):
            return self.b2(self.b1(x))

    return _Deep()
