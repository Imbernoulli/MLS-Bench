"""Agent-editable alpha-refinement surface for full-inventory image matting.

Keep refine(coarse_alpha, image, trimap). The selected function is executed
directly during training and evaluation; invalid output is never replaced.
"""
from __future__ import annotations

import torch


# EDITABLE REGION
def refine(coarse_alpha, image, trimap):
    # Native identity refinement.
    return coarse_alpha
# END EDITABLE REGION
