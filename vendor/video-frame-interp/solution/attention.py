"""Agent-editable surface: ATTENTION in the refinement bottleneck for video frame interpolation.

A FIXED harness trains the SAME interpolator on tiny two-layer (occluding) triplets and scores
interpolation PSNR over three inter-frame MOTION MAGNITUDES. You design ONLY the feature
aggregation block at the bottleneck of the refinement/synthesis U-Net; everything else is
fixed.

    def get_attention_config():
        return {'kind': 'nonlocal'}

`kind` chooses the bottleneck aggregation:
  'none'     -> plain conv bottleneck (local receptive field only). (weak floor)
  'se'       -> a squeeze-and-excitation CHANNEL gate re-weights features globally. (mid)
  'nonlocal' -> a spatial SELF-ATTENTION (non-local) block lets the bottleneck aggregate
                content from far-away visible regions to fill disocclusion -> highest PSNR.
                (strong / SOTA)

Known ordering (PSNR): none < se < nonlocal, WIDENING with motion (long-range aggregation helps
wider disocclusion). The DEFAULT below returns 'none'. A malformed / crashing return falls back
to 'nonlocal'.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION -- design the attention surface below
# ================================================================
def get_attention_config():
    # DEFAULT: the WEAK baseline -- improve this toward the SOTA choice.
    return {'kind': 'none'}
# ================================================================
# END EDITABLE REGION
# ================================================================
