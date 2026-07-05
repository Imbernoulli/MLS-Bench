"""Agent-editable surface: OPTICAL-FLOW ESTIMATION module for video frame interpolation.

A FIXED harness trains the SAME Super-SloMo-style interpolator (backward-warp both frames to
t=0.5, learned visibility mask + refinement) a few hundred steps on tiny fixed two-layer
(occluding) (frame0, frame2, true-middle) triplets, and scores interpolation PSNR of the
synthesized middle frame vs the true middle frame, over three inter-frame MOTION MAGNITUDES.
You design ONLY the flow-estimation module that feeds the warp; everything else is fixed.

    def get_flow_config():
        return {'kind': 'refine'}

`kind` chooses HOW the bidirectional flow to t=0.5 is estimated:
  'zero'   -> no flow at all (flow=0). The warp degenerates to a plain average of the two
              frames -> ghosts as motion grows. (weak floor)
  'single' -> a single one-shot flow net predicts F_{t->0}, F_{t->2} in one pass. Compensates
              bulk motion but leaves residual flow error at (dis)occlusion boundaries. (mid)
  'refine' -> coarse flow + RAFT-style iterative residual-flow refinement on the warped inputs
              -> recovers the large occluding motion. Highest PSNR. (strong / SOTA)

Known ordering (PSNR): zero < single < refine, WIDENING with motion (zero collapses fastest).
The DEFAULT below returns 'zero' (the weak, no-motion baseline). A malformed / crashing /
unknown return falls back to the SOTA 'refine'.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION -- design the flow surface below
# ================================================================
def get_flow_config():
    # DEFAULT: the WEAK baseline -- improve this toward the SOTA choice.
    return {'kind': 'zero'}
# ================================================================
# END EDITABLE REGION
# ================================================================
