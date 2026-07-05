"""Agent-editable surface: NETWORK DEPTH / ARCHITECTURE of the mask-guided deshadower.

A FIXED training protocol (mask-guided residual learning, robust L1 + SSIM loss up-weighted
in the shadow, fixed optimiser/iters/seed) removes a cast shadow (physics-based linear
illumination model  I = a*J,  a = 1 - (1-att)*m, SP+M-Net, Le et al. ICCV 2019). You design
ONLY the depth of the U-Net encoder-decoder:

    def get_arch_config():
        # {'depth': 1 | 2}
        return {'depth': 2}

  * depth=1 -> a SHALLOW 1-level U-Net (one 2x downsampling stage). Small receptive field:
               it cannot see the whole soft shadow at once, so it under-corrects large
               umbrae -> lower shadow-region PSNR. WEAK.
  * depth=2 -> the DEEPER 2-level encoder-decoder (two downsampling stages) -> a larger
               receptive field that covers big soft shadows and models the smooth penumbra
               falloff -> higher shadow-region PSNR. The strong answer.

The DEFAULT below returns depth=1 (shallow). Adding the second level clearly raises shadow-
region PSNR (most on the harder heavy setting, where shadows are larger/darker). A malformed
/ crashing return falls back to depth=1 (weak).
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — design the architecture (depth) config below
# ================================================================
def get_arch_config():
    # Default: shallow 1-level U-Net (small receptive field).
    return {"depth": 1}
# ================================================================
# END EDITABLE REGION
# ================================================================
