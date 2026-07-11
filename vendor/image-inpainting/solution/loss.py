"""Agent-editable solution surface for image inpainting.

Keep the required callable names and signatures below. Follow the fixed harness
contract for return types, shapes, devices, value ranges, finiteness, and
determinism. The selected implementation is evaluated directly; contract or
runtime failures invalidate the run.
"""
from __future__ import annotations

import torch


# EDITABLE REGION
def compute_loss(out, gt, mask):
    valid = 1.0 - mask
    return (torch.abs(out - gt) * valid).sum() / (valid.sum() + 1e-8)
# END EDITABLE REGION
