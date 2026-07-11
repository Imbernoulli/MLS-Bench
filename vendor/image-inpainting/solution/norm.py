"""Agent-editable solution surface for image inpainting.

Keep the required callable names and signatures below. Follow the fixed harness
contract for return types, shapes, devices, value ranges, finiteness, and
determinism. The selected implementation is evaluated directly; contract or
runtime failures invalidate the run.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# EDITABLE REGION
def make_norm(num_ch):
    return nn.BatchNorm2d(num_ch)
# END EDITABLE REGION
