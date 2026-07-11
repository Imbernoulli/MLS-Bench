"""Agent-editable bottleneck channel-gating surface for image matting.

Keep build_attention(ch). The returned module maps a BxCxHxW bottleneck to finite
BxCx1x1 gates in [0,1]. The selected implementation is evaluated directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# EDITABLE REGION — design the bottleneck channel gate below
# ================================================================
def build_attention(ch):
    # Native parameter-free global-mean gate.
    class MeanGate(nn.Module):
        def forward(self, x):
            return torch.sigmoid(x.mean(dim=(-2, -1), keepdim=True))
    return MeanGate()
# ================================================================
# END EDITABLE REGION
# ================================================================
