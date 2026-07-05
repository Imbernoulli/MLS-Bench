"""Agent-editable surface: the BOTTLENECK DILATION / RECEPTIVE-FIELD block (dilation).

Return a torch.nn.Module via build_dilation(ch) whose forward maps a bottleneck
feature (B, ch, H, W) to the SAME shape. It is inserted at the stride-8 bottleneck
(after the attention block). Everything else is FIXED; only this block changes.
Scored by SAD (LOWER is better) in the trimap UNKNOWN band, gmean over three
trimap-width settings.

A dilated multi-rate block enlarges the receptive field WITHOUT losing resolution,
aggregating context across the wide unknown band (ASPP, Chen et al. 2017; dilated
context, Iizuka et al. 2017). Order on this data:
    single 3x3 (limited context)  <  pointwise 1x1 (channel mixing only)
      <  dilated multi-rate block (best context aggregation = SOTA).
(Wait — a single 3x3 has MORE spatial context than a 1x1. The measured order here is
that a bare single 3x3 conv is a weak default, a pointwise 1x1 is even weaker context
but a fine residual, and a DILATED multi-rate block that fuses rates 1/2/4 gives the
strongest context. The default is the weak single 3x3.)

The DEFAULT below is a deliberately weak SINGLE 3x3 conv (limited receptive field, no
multi-scale context). Redesign build_dilation() as a DILATED multi-rate residual
block (parallel dilations 1/2/4, fused) that aggregates wide context across the
unknown band, with clear headroom. A malformed / crashing / wrong-shape module falls
back to identity.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# EDITABLE REGION — design the bottleneck dilation/context block below
# ================================================================
def build_dilation(ch):
    # Default: a SINGLE plain 3x3 conv. Limited receptive field, no multi-scale
    # context aggregation -> the wide unknown band is under-contextualised -> higher
    # SAD. A dilated multi-rate block (rates 1/2/4, fused) aggregates far more context.
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
