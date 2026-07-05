"""Agent-editable surface: FEATURE FUSION -- what feeds the refinement/synthesis net.

A FIXED harness trains the SAME interpolator on tiny two-layer (occluding) triplets and scores
interpolation PSNR over three inter-frame MOTION MAGNITUDES. You design ONLY which signals are
FUSED as input to the refinement U-Net (which predicts the visibility mask + residual);
everything else is fixed.

    def get_fusion_config():
        return {'kind': 'full'}

`kind` chooses the fusion input:
  'warps'     -> only the two warped RGB candidates. The net cannot see WHERE the flow is
                 uncertain -> weak occlusion reasoning. (weak floor)
  'plus_flow' -> warped candidates + the two flow fields (the net can reason about motion
                 magnitude / disocclusion). (mid)
  'full'      -> warped candidates + flows + the ORIGINAL frames + warped CONTEXT features ->
                 richest fusion, best disocclusion synthesis. Highest PSNR. (strong / SOTA)

Known ordering (PSNR): warps < plus_flow < full, WIDENING with motion. The DEFAULT below
returns 'warps'. A malformed / crashing return falls back to 'full'.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION -- design the fusion surface below
# ================================================================
def get_fusion_config():
    # DEFAULT: the WEAK baseline -- improve this toward the SOTA choice.
    return {'kind': 'warps'}
# ================================================================
# END EDITABLE REGION
# ================================================================
