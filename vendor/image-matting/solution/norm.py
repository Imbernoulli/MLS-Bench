"""Agent-editable normalization surface for full-inventory image matting.

Keep make_norm(num_ch). The returned torch.nn.Module must preserve feature shape and
produce finite output. The selected implementation is evaluated directly.
"""
from __future__ import annotations

import torch.nn as nn


# ================================================================
# EDITABLE REGION — design the normalisation layer below
# ================================================================
def make_norm(num_ch):
    # Native identity implementation.
    return nn.Identity()
# ================================================================
# END EDITABLE REGION
# ================================================================
