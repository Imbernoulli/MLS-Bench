"""Agent-editable receptive-field surface for full-inventory image matting.

Keep build_dilation(ch). The returned module must preserve the complete input shape
and produce finite output. The selected implementation is evaluated directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# EDITABLE REGION — design the bottleneck dilation/context block below
# ================================================================
def build_dilation(ch):
    # Native residual 3x3 implementation.
    class SingleConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(ch, ch, 3, padding=1)
            self.bn = nn.BatchNorm2d(ch)

        def forward(self, x):
            return x + F.relu(self.bn(self.conv(x)))
    return SingleConv()
# ================================================================
# END EDITABLE REGION
# ================================================================
