"""Agent-editable loss surface for full-inventory image matting.

Keep get_matting_loss(). The returned callable must produce one finite non-negative
scalar tensor with a gradient path. The selected implementation is evaluated
directly.
"""
from __future__ import annotations

import torch


# ================================================================
# EDITABLE REGION — design the alpha-matte training loss below
# ================================================================
def get_matting_loss():
    # Native whole-image alpha-L1 implementation.
    def loss_fn(pred, gt, image, fg, bg, trimap, unknown):
        return (pred - gt).abs().mean()
    return loss_fn
# ================================================================
# END EDITABLE REGION
# ================================================================
