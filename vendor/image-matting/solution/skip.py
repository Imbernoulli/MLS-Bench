"""Agent-editable skip-fusion surface for full-inventory image matting.

Keep fuse(dec_up, skip). The function must return a finite tensor with the exact
batch, summed-channel, and spatial shape required by the fixed decoder.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ================================================================
# EDITABLE REGION — design the skip fusion below
# ================================================================
def fuse(dec_up, skip):
    # Native zero-skip implementation that preserves the required channel count.
    return torch.cat([dec_up, torch.zeros_like(skip)], 1)
# ================================================================
# END EDITABLE REGION
# ================================================================
