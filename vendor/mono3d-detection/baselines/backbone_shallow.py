"""mono3d-head-capacity WEAK baseline: SHALLOW/NARROW refinement (near-identity).

Refine the shared embedding with a tiny bottleneck (emb -> 8 -> emb, no residual). Squeezing the
128-d embedding through an 8-d bottleneck DESTROYS most of the encoded depth/pose information
before it reaches the task heads -> a severe capacity/information bottleneck, lower AP3D. This is
the deliberately under-capacity reference. Reference: standard capacity ablation — too-narrow a
refinement block bottlenecks the representation.
"""
import torch.nn as nn


def build_backbone(emb_dim):
    class _Narrow(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(emb_dim, 8), nn.ReLU(), nn.Linear(8, emb_dim))

        def forward(self, x):
            return self.f(x)                 # NO residual: the 8-d bottleneck is a hard squeeze

    return _Narrow()
