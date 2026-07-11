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
def build_dilation(ch):
    class SingleConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(ch, ch, 3, 1, 1, dilation=1)
            self.act = nn.ReLU(True)

        def forward(self, x):
            return self.act(self.c(x))

    return SingleConv()
# END EDITABLE REGION
