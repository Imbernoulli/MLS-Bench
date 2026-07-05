"""Agent-editable surface: MASK GUIDANCE for the image shadow-remover.

A FIXED training protocol (residual learning, a robust L1 + SSIM loss up-weighted in the
shadow region, a fixed optimiser/iters/seed) trains a mask-guided residual deshadower to
recover a clean, shadow-free image from one on which a CAST SHADOW multiplicatively darkens a
known, soft-edged region (the physics-based linear illumination model  I = a*J,
a = 1 - (1-att)*m  of Shadow Image Decomposition / SP+M-Net, Le et al. ICCV 2019). You design
ONLY whether the soft shadow MASK m is fed to the net as a 4th input channel:

    def get_mask_config():
        # {'use_mask': True | False}
        return {'use_mask': True}

  * use_mask=False -> a BLIND U-Net that sees only the 3-channel shadowed RGB and must both
                      LOCATE and correct the shadow from colour alone (DeshadowNet-style,
                      Qu et al. CVPR 2017, WITHOUT the mask prior). It leaks into the lit
                      region and mis-corrects the soft penumbra -> lower shadow-region PSNR.
  * use_mask=True  -> the MASK-GUIDED U-Net: the soft shadow mask is concatenated as a 4th
                      input channel, so the net is told exactly WHERE and HOW MUCH to
                      brighten -- the SP+M-Net physically-parameterised recovery. Higher
                      shadow-region PSNR.

The DEFAULT below returns use_mask=False (the blind net). Feeding the mask gives clear
shadow-region PSNR headroom. A malformed / crashing return falls back to use_mask=False.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — design the mask-guidance config below
# ================================================================
def get_mask_config():
    # Default: blind U-Net (no mask channel).
    return {"use_mask": False}
# ================================================================
# END EDITABLE REGION
# ================================================================
