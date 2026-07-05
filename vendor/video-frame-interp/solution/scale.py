"""Agent-editable surface: FLOW-ESTIMATION SCALE / coarse-to-fine PYRAMID.

A FIXED harness trains the SAME interpolator on tiny two-layer (occluding) triplets and scores
interpolation PSNR over three inter-frame MOTION MAGNITUDES. You design ONLY the number of
scales in the flow-estimation pyramid (a coarse-to-fine flow net captures larger motion);
everything else is fixed.

    def get_scale_config():
        return {'levels': 'three'}

`levels` chooses the flow-net pyramid depth:
  'single' -> full-resolution flow net only (1 scale). Small receptive field -> cannot capture
              large motion -> collapses as motion grows. (weak floor)
  'two'    -> a 2-level coarse-to-fine pyramid. Captures medium motion. (mid)
  'three'  -> a 3-level pyramid -> captures the large occluding motion. Highest PSNR. (strong /
              SOTA)

Known ordering (PSNR): single < two < three, WIDENING with motion (deeper pyramid needed for
larger displacement). The DEFAULT below returns 'single'. A malformed / crashing return falls
back to 'three'.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION -- design the scale surface below
# ================================================================
def get_scale_config():
    # DEFAULT: the WEAK baseline -- improve this toward the SOTA choice.
    return {'levels': 'single'}
# ================================================================
# END EDITABLE REGION
# ================================================================
