"""Agent-editable surface: REFINEMENT / SYNTHESIS NETWORK DEPTH for video frame interpolation.

A FIXED harness trains the SAME interpolator (learned flow + warp) on tiny two-layer
(occluding) triplets and scores interpolation PSNR over three inter-frame MOTION MAGNITUDES.
You design ONLY the DEPTH of the refinement/synthesis U-Net that predicts the visibility mask
and residual from the two warped candidates; everything else is fixed.

    def get_refine_config():
        return {'depth': 'deep'}

`depth` chooses the refinement-net capacity:
  'none'    -> NO refinement net: just a fixed 0.5 average of the two warped candidates ->
               ghosts at disocclusion. (weak floor)
  'shallow' -> a 1-level refinement U-Net predicts the mask + residual. Some occlusion
               reasoning, limited receptive field. (mid)
  'deep'    -> the full 3-level refinement U-Net -> resolves wide disocclusion. Highest PSNR.
               (strong / SOTA)

Known ordering (PSNR): none < shallow < deep, WIDENING with motion (a deeper net sees wider
disocclusion). The DEFAULT below returns 'none'. A malformed / crashing return falls back to
'deep'.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION -- design the refine surface below
# ================================================================
def get_refine_config():
    # DEFAULT: the WEAK baseline -- improve this toward the SOTA choice.
    return {'depth': 'none'}
# ================================================================
# END EDITABLE REGION
# ================================================================
