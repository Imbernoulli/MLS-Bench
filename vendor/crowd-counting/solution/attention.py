"""Agent-editable surface: spatial ATTENTION on the backbone features.

Define `build_attention(cin)` -> a torch.nn.Module mapping features `(B, cin, h, w)` to
gated features of the same shape. It runs between the frontend and the density tail;
only the attention module changes.

Without attention (identity), the unannotated distractor CLUTTER in the background is
not suppressed, so the counter spends density mass on non-objects and mis-counts ->
higher counting MAE. A learned spatial-attention gate predicts a per-pixel weight in
[0,1] and multiplies the features by it, suppressing clutter and focusing on real
objects -> lower MAE. This mirrors attention-based counters (SCAR, ADCrowdNet, SFANet).

    def build_attention(cin):
        import torch.nn as nn
        class SpatialAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Sequential(
                    nn.Conv2d(cin, cin // 2, 3, padding=1), nn.ReLU(True),
                    nn.Conv2d(cin // 2, 1, 1), nn.Sigmoid())
            def forward(self, x): return x * self.gate(x)
        return SpatialAttention()

The DEFAULT below is the deliberately weak NO-attention identity. A crashing / malformed
attention module falls back to identity.
"""
from __future__ import annotations

import torch.nn as nn


# ================================================================
# EDITABLE REGION — design the spatial attention module below
# ================================================================
def build_attention(cin):
    # Default: NO attention (weak). Distractor clutter not suppressed -> higher MAE.
    return nn.Identity()
# ================================================================
# END EDITABLE REGION
# ================================================================
