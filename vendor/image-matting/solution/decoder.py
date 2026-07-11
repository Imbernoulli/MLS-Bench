"""Agent-editable decoder surface for full-inventory image matting.

Keep build_decoder(enc_channels). The returned module consumes the fixed encoder
feature list and must produce a finite full-resolution alpha tensor in [0,1].
The selected implementation is evaluated directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# EDITABLE REGION — design the matting decoder below
# ================================================================
def build_decoder(enc_channels):
    # Native deepest-feature projection implementation.
    c0, c1, c2, c3 = enc_channels

    class Dec(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(nn.Conv2d(c3, 32, 3, padding=1), nn.ReLU(True),
                                      nn.Conv2d(32, 1, 1))

        def forward(self, feats):
            e3 = feats[-1]
            a = self.proj(e3)
            a = F.interpolate(a, size=feats[0].shape[-2:], mode="bilinear",
                              align_corners=False)
            return torch.sigmoid(a).squeeze(1)
    return Dec()
# ================================================================
# END EDITABLE REGION
# ================================================================
