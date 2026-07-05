"""Agent-editable surface: the DENSITY-HEAD SPATIAL AGGREGATION (normalisation).

Return a torch.nn.Module `head` on the frontend features `(B, cin, h, w)` that outputs
a NON-NEGATIVE per-pixel DENSITY MAP `(B, h, w)`; the object count is always its
spatial INTEGRAL. The design choice is HOW the head produces that map:

  - FREE density field: a convolutional head with a non-negative activation
    (softplus/relu) per pixel. The total integrated mass is UNBOUNDED, so the count
    can grow to match arbitrarily crowded scenes -> it EXTRAPOLATES to higher counts.

  - GLOBALLY-NORMALISED distribution x scalar: a spatial SOFTMAX (the map sums to 1
    per image, a pure *distribution* of WHERE objects are) multiplied by a single
    learned/pooled COUNT SCALAR (HOW MANY). This factorises location from count, but
    the total mass is bottlenecked through ONE scalar; trained on LOW-count images the
    scalar saturates near the training mean and CANNOT scale up -> it fails to
    extrapolate to higher-count val images, just like a direct count regressor.

    def build_density_head(cin):
        import torch.nn as nn, torch.nn.functional as F
        class Head(nn.Module):                          # FREE density field
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(cin, 64, 3, padding=1), nn.ReLU(True),
                    nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(True),
                    nn.Conv2d(32, 1, 1))
            def forward(self, feat):
                return F.softplus(self.net(feat)).squeeze(1)      # (B,h,w) density
        return Head()

The DEFAULT below is the deliberately weak GLOBALLY-NORMALISED head: a spatial softmax
(mass = 1) times a learned count scalar. Its total mass is bottlenecked through the
scalar, which saturates at the low training mean and cannot represent the higher val
counts -> high counting MAE. Switching to a FREE non-negative density field lets the
integral scale to any count and recovers accuracy with clear headroom. A malformed /
crashing head falls back to the default free density head.

The count scale is fixed at 100.0 (the harness DENSITY_SCALE): the GT density map
integrates to 100 * count, so a correct head's map integrates to ~100 * count.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# EDITABLE REGION — design the density head's spatial aggregation below
# ================================================================
def build_density_head(cin):
    # Default: GLOBALLY-NORMALISED distribution x learned scalar (weak). The spatial
    # softmax makes the map a pure distribution (sums to 1); a single learned scalar
    # sets the total count. The scalar saturates at the low train mean -> cannot
    # extrapolate to higher val counts -> high counting MAE.
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, 64, 3, padding=1), nn.ReLU(True),
                nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(True),
                nn.Conv2d(32, 1, 1))
            self.count_scalar = nn.Parameter(torch.tensor(50.0))

        def forward(self, feat):
            m = self.net(feat)                       # (B,1,h,w)
            B, _, h, w = m.shape
            dist = F.softmax(m.view(B, -1), dim=1).view(B, h, w)   # sums to 1 per image
            total = F.softplus(self.count_scalar) * 100.0          # DENSITY_SCALE=100
            return dist * total                      # (B,h,w) density, mass = total
    return Head()
# ================================================================
# END EDITABLE REGION
# ================================================================
