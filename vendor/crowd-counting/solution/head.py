"""Agent-editable surface: the COUNT PREDICTION HEAD (the counting FORMULATION).

Return a torch.nn.Module `head` on the frontend features `(B, cin, h, w)` at stride 8.
The head decides the CORE crowd-counting formulation, and the harness scores whichever
it returns:
  - a NON-NEGATIVE DENSITY MAP `(B, h, w)` -> the count is its SPATIAL INTEGRAL
    (the density formulation: MCNN / CSRNet). ~h*w spatially-local predictions are
    summed, each densely supervised, so the per-image count is a LOW-VARIANCE estimate.
  - a per-image SCALAR `(B,)` -> used DIRECTLY as the count (direct global regression).
    A single number is regressed from a globally-pooled feature with ONE supervision
    signal per image and no spatial inductive bias, so it overfits the count
    distribution (regresses toward the training mean) and has HIGH per-image variance.

    def build_count_head(cin):
        import torch.nn as nn, torch.nn.functional as F
        class Head(nn.Module):                    # DENSITY-MAP formulation
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(cin, 64, 3, padding=1), nn.ReLU(True),
                    nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(True),
                    nn.Conv2d(32, 1, 1))
            def forward(self, feat):
                return F.softplus(self.net(feat)).squeeze(1)   # (B,h,w) density
        return Head()

The DEFAULT below is the deliberately weak DIRECT GLOBAL COUNT REGRESSION head: it
global-average-pools the features and maps them through an MLP to a single scalar
count. This throws away the spatial density structure that makes counting robust, so
it regresses toward the mean count and generalises poorly -> high counting MAE.
Switching to the DENSITY-MAP formulation (predict a non-negative per-pixel density and
integrate it) recovers counting accuracy with clear headroom -- this is the founding
result of density-based counting. A malformed / crashing head falls back to the
default density head.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# EDITABLE REGION — design the count prediction head below
# ================================================================
def build_count_head(cin):
    # Default: DIRECT global count regression (weak). Global-average-pool -> MLP ->
    # ONE scalar count. No spatial density -> high-variance, regresses to the mean.
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(cin, 128), nn.ReLU(True),
                nn.Linear(128, 64), nn.ReLU(True),
                nn.Linear(64, 1))

        def forward(self, feat):
            pooled = feat.mean(dim=(-2, -1))          # (B, cin) global average pool
            return F.softplus(self.mlp(pooled)).squeeze(-1)   # (B,) scalar count
    return Head()
# ================================================================
# END EDITABLE REGION
# ================================================================
