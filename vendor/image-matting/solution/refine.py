"""Agent-editable surface: the SECOND-STAGE ALPHA REFINEMENT (refine).

Return a function
    refine(coarse_alpha, x, trimap) -> refined_alpha (B,H,W) in [0,1]
where
    coarse_alpha (B,H,W)  the alpha from a FIXED coarse matting stage (U-Net)
    x            (B,C,H,W) the network input = concat(RGB, trimap-encoding)
    trimap       (B,H,W)  the trimap in {0,0.5,1}
A fixed coarse ConfigMattingNet produces a first-pass matte; your refine() is a
SECOND stage that sharpens it. Everything else (data, encoder, decoder, trimap,
loss, optimiser, iterations, seed, eval) is FIXED. Scored by SAD (LOWER is better)
in the trimap UNKNOWN band, gmean over three trimap-width settings.

Deep Image Matting (Xu et al. 2017) uses a SECOND refinement network that takes the
image + the coarse alpha and predicts a residual correction, giving a sharper matte
along the transition. The order is:
    identity (no refinement)  <  a shallow refine net  <  a full residual refinement
    stage (Xu 2017 = SOTA).

The DEFAULT below is a deliberately weak IDENTITY (single-pass): it returns the
coarse alpha unchanged -> no refinement -> the coarse matte's residual error along
the transition is never corrected. Redesign refine() as a PARAMETER-FREE second stage
that sharpens the coarse matte (e.g. unsharp masking to counter the coarse blur, plus
snapping the DEFINITE fg/bg trimap regions to their known 1/0 values so only the
unknown band carries soft values). A malformed / crashing refine falls back to
identity.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# EDITABLE REGION — design the second-stage alpha refinement below
# ================================================================
def refine(coarse_alpha, x, trimap):
    # Default: IDENTITY (single-pass). Return the coarse matte unchanged -> no
    # refinement -> the coarse residual error along the soft transition is left
    # uncorrected. A real second stage predicts a bounded residual and sharpens.
    return coarse_alpha
# ================================================================
# END EDITABLE REGION
# ================================================================
