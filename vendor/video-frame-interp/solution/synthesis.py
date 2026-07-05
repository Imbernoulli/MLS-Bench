"""Agent-editable surface: MIDDLE-FRAME SYNTHESIS STRATEGY for video frame interpolation.

A FIXED harness trains a compact interpolation model a few hundred steps on TINY fixed sets
of (frame0, frame2, true-middle-frame) triplets -- synthetic frames warped along a KNOWN
motion, so the middle frame at t=0.5 is EXACT -- and scores the PSNR of the SYNTHESIZED
middle frame vs the true middle frame, aggregated over three inter-frame MOTION MAGNITUDES.
You design ONLY the synthesis strategy.

    def get_synthesis_config():
        return {'method': 'learned'}

`method` chooses HOW the middle frame is built:
  'blend'     -> naive linear blend 0.5*(frame0 + frame2). Ignores motion entirely, so it
                 GHOSTS / doubles moving edges; PSNR collapses as motion grows. (weak floor)
  'flow_warp' -> estimate the bidirectional flow to t=0.5 with a small learnable flow net,
                 BACKWARD-WARP frame0 and frame2 to the middle time, and average the two
                 motion-compensated candidates. Compensates motion -> much higher PSNR than
                 blend, but uses a fixed 0.5 occlusion/blend. (mid)
  'learned'   -> flow_warp PLUS a refinement/synthesis U-Net that predicts a soft per-pixel
                 visibility mask to combine the two warped candidates and adds a residual
                 correction -- the Super-SloMo (Jiang et al., CVPR 2018) flow-computation +
                 flow-interpolation + refinement design. Highest PSNR. (strong / SOTA)

Known ordering (PSNR): blend < flow_warp < learned, WIDENING with motion (blend collapses
fastest). The DEFAULT below returns 'blend' -> the weak motion-agnostic baseline. Returning
'learned' gives clear PSNR headroom. A malformed / crashing return falls back to 'learned'.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — design the synthesis strategy below
# ================================================================
def get_synthesis_config():
    # Default: naive linear blend (no motion) -> ghosts as motion grows, lowest PSNR.
    return {"method": "blend"}
# ================================================================
# END EDITABLE REGION
# ================================================================
