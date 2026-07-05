"""Agent-editable surface: DECODER UPSAMPLING of the mask-guided deshadower.

A FIXED mask-guided residual deshadower (same U-Net width/depth, loss, optimiser, data,
iters, seed, eval split) removes a cast shadow (SP+M-Net linear illumination model, Le et al.
ICCV 2019). You design ONLY the decoder upsampler:

    def get_upsampling_config():
        # {'up': 'transpose' | 'bilinear'}
        return {'up': 'bilinear'}

  * up='transpose' -> transpose-convolution (deconv) upsampling. Prone to CHECKERBOARD
                      artifacts across the smooth soft shadow -> lower shadow-region PSNR.
  * up='bilinear'  -> BILINEAR-resize + conv upsampling. Smooth, artifact-free upsampling
                      that respects the soft penumbra falloff -> higher shadow-region PSNR.

The DEFAULT below returns up='transpose'. The resize-conv upsampler avoids checkerboard
artifacts on the smooth shadow and typically raises shadow-region PSNR. A malformed /
crashing return falls back to up='transpose' (weak).
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — design the upsampling config below
# ================================================================
def get_upsampling_config():
    # Default: transpose-conv (checkerboard-prone) upsampling.
    return {"up": "transpose"}
# ================================================================
# END EDITABLE REGION
# ================================================================
