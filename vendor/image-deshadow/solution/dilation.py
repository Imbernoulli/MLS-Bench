"""Agent-editable surface: DILATION / RECEPTIVE FIELD of the mask-guided deshadower.

A FIXED mask-guided residual deshadower (same U-Net width/depth, loss, optimiser, data,
iters, seed, eval split) removes a cast shadow (SP+M-Net linear illumination model, Le et al.
ICCV 2019). You design ONLY the per-block DILATION schedule of the bottleneck ResBlocks,
which sets the receptive field:

    def get_dilation_config():
        # {'dilations': [d1, d2]}  (dilation rate of the two bottleneck conv blocks)
        return {'dilations': [2, 4]}

  * dilations=[1, 1] -> plain 3x3 convs, SMALL receptive field. The net cannot see the full
                        extent of a large soft shadow in one pass -> under-corrects the
                        umbra centre -> lower shadow-region PSNR. WEAK.
  * dilations=[2, 4] -> DILATED bottleneck convs -> a much LARGER receptive field (à la
                        multi-context deshadowing / ASPP) that covers big shadows and models
                        the smooth penumbra falloff -> higher shadow-region PSNR.

The DEFAULT below returns dilations=[1, 1]. The dilated trunk helps most on the larger HEAVY
shadows. A malformed / crashing return falls back to dilations=[1, 1] (weak).
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — design the dilation schedule below
# ================================================================
def get_dilation_config():
    # Default: no dilation (small receptive field).
    return {"dilations": [1, 1]}
# ================================================================
# END EDITABLE REGION
# ================================================================
