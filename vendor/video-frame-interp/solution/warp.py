"""Agent-editable surface: MOTION-COMPENSATED WARPING for video frame interpolation.

A FIXED harness trains the SAME interpolator (learned flow + visibility mask + refinement) on
tiny two-layer (occluding) triplets and scores interpolation PSNR of the synthesized middle
frame, over three inter-frame MOTION MAGNITUDES. You design ONLY how the two frames are
motion-compensated (warped) to t=0.5; everything else is fixed.

    def get_warp_config():
        return {'kind': 'softsplat'}

`kind` chooses the warping operator:
  'none'      -> no warp (use the frames as-is). Equivalent to a blend -> ghosts. (weak floor)
  'forward'   -> FORWARD warp / splatting: push each frame's pixels along the flow to t=0.5 and
                 accumulate. Motion-correct but leaves HOLES where nothing lands. (weak-mid)
  'backward'  -> BACKWARD (inverse) warp: sample each frame at grid+flow. Dense, no holes, the
                 standard VFI warp. (mid)
  'softsplat' -> SOFTMAX-SPLATTING (Niklaus & Liu, CVPR 2020): forward-splat with importance
                 weights AND backfill remaining holes from the backward warp -> the best warp,
                 resolves disocclusion. Highest PSNR. (strong / SOTA)

Known ordering (PSNR): none < forward < backward < softsplat, WIDENING with motion. The
DEFAULT below returns 'none'. A malformed / crashing return falls back to 'softsplat'.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION -- design the warp surface below
# ================================================================
def get_warp_config():
    # DEFAULT: the WEAK baseline -- improve this toward the SOTA choice.
    return {'kind': 'none'}
# ================================================================
# END EDITABLE REGION
# ================================================================
