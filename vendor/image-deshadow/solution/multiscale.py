"""Agent-editable surface: MULTI-SCALE (coarse-to-fine) deshadowing.

A FIXED mask-guided residual deshadower (same per-branch U-Net width, loss, optimiser, data,
iters, seed, eval split) removes a cast shadow (SP+M-Net linear illumination model, Le et al.
ICCV 2019). You design ONLY whether the net is single-scale or a COARSE-TO-FINE pyramid:

    def get_multiscale_config():
        # {'multiscale': True | False}
        return {'multiscale': True}

  * multiscale=False -> a single-scale mask-guided U-Net. WEAK.
  * multiscale=True  -> a COARSE-TO-FINE pyramid: the input (+ mask) is relit at HALF
                        resolution first (a large effective receptive field that captures
                        the WHOLE big soft shadow) and that coarse estimate is fused into a
                        full-resolution refinement branch (multi-scale deshadow, cf. MSPFN /
                        pyramid restoration) -> higher shadow-region PSNR on large shadows.

The DEFAULT below returns multiscale=False. The pyramid helps most on the larger HEAVY
shadows. A malformed / crashing return falls back to multiscale=False (weak).
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — design the multiscale config below
# ================================================================
def get_multiscale_config():
    # Default: single-scale (no coarse-to-fine pyramid).
    return {"multiscale": False}
# ================================================================
# END EDITABLE REGION
# ================================================================
