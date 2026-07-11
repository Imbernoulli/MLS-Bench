"""Agent-editable solution surface for image inpainting.

Keep the required callable names and signatures below. Follow the fixed harness
contract for return types, shapes, devices, value ranges, finiteness, and
determinism. The selected implementation is evaluated directly; contract or
runtime failures invalidate the run.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# EDITABLE REGION
def make_activation():
    return nn.LeakyReLU(negative_slope=0.2, inplace=True)
# END EDITABLE REGION
