"""Agent-editable upsampling surface for full-inventory image matting.

Keep build_upsampler(cin). The returned module must map (B,cin,H,W) to the exact
finite shape (B,cin,2H,2W). The harness does not repair malformed active output.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# EDITABLE REGION — design the decoder upsampling operator below
# ================================================================
def build_upsampler(cin):
    # Native nearest-neighbor implementation.
    class NearestUp(nn.Module):
        def forward(self, x):
            return F.interpolate(x, scale_factor=2, mode="nearest")
    return NearestUp()
# ================================================================
# END EDITABLE REGION
# ================================================================
