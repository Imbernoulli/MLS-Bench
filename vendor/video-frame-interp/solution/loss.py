"""Agent-editable surface: TRAINING LOSS / OBJECTIVE for video frame interpolation.

A FIXED harness trains the SAME interpolator (learned flow + mask + refinement) on tiny
two-layer (occluding) triplets and scores interpolation PSNR over three inter-frame MOTION
MAGNITUDES. You design ONLY the training objective; the model and optimiser are fixed.

    def get_loss_config():
        return {'kind': 'l1_warp'}

`kind` chooses the loss:
  'l2'        -> plain MSE. Over-smooths edges -> blurs the moving boundary. (weak floor)
  'l1'        -> Charbonnier (robust L1). Sharper than L2. (mid)
  'l1_census' -> Charbonnier + a census/edge (image-gradient) term -> sharpens the
                 (dis)occlusion boundary. (mid-strong)
  'l1_warp'   -> Charbonnier + census + a WARP self-consistency term (each warped candidate
                 must also match GT) -> best supervision of the occluded region. Highest PSNR.
                 (strong / SOTA)

Known ordering (PSNR): l2 < l1 < l1_census < l1_warp, WIDENING with motion (sharper losses
help more where disocclusion is wide). The DEFAULT below returns 'l2'. A malformed / crashing
return falls back to 'l1_warp'.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION -- design the loss surface below
# ================================================================
def get_loss_config():
    # DEFAULT: the WEAK baseline -- improve this toward the SOTA choice.
    return {'kind': 'l2'}
# ================================================================
# END EDITABLE REGION
# ================================================================
