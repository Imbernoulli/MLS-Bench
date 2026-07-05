"""Agent-editable surface: NUMBER OF FLOW-REFINEMENT ITERATIONS (RAFT-style).

A FIXED harness trains the SAME interpolator on tiny two-layer (occluding) triplets and scores
interpolation PSNR over three inter-frame MOTION MAGNITUDES. You design ONLY how many iterative
residual-flow refinement steps the flow estimator runs (RAFT, Teed & Deng 2020); everything
else is fixed.

    def get_flow_iters_config():
        return {'n': 4}

`n` chooses the number of flow-refinement iterations (allowed: 1, 2, 4):
  1 -> a single flow pass, no iterative refinement -> residual flow error survives at large
       occluding motion. (weak floor)
  2 -> two refinement iterations -> recovers medium motion. (mid)
  4 -> four refinement iterations -> recovers the large occluding motion. Highest PSNR.
       (strong / SOTA)

Known ordering (PSNR): 1 < 2 < 4, WIDENING with motion (more iterations for larger
displacement). The DEFAULT below returns 1. A malformed / crashing / out-of-range return falls
back to 4.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION -- design the iters surface below
# ================================================================
def get_flow_iters_config():
    # DEFAULT: the WEAK baseline -- improve this toward the SOTA choice.
    return {'n': 1}
# ================================================================
# END EDITABLE REGION
# ================================================================
