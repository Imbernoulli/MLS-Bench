"""Agent-editable surface: DESHADOWER BACKBONE for the image shadow-remover.

A FIXED training protocol (residual learning, a robust L1 + SSIM loss up-weighted in the
shadow region, a fixed optimiser/iters/seed) trains a deshadower to recover a clean,
shadow-free image from one on which a CAST SHADOW multiplicatively darkens a known, soft-
edged region (the physics-based linear illumination model  I = a*J,  a = 1 - (1-att)*m  of
Shadow Image Decomposition / SP+M-Net, Le et al. ICCV 2019). The SOFT SHADOW MASK m is
available as an extra input. You design ONLY the network BACKBONE and whether it uses the
mask:

    def get_network_config():
        # {'arch': 'copy' | 'unet_nomask' | 'unet_mask'}
        return {'arch': 'unet_mask'}

  * 'copy'        -> pass the shadowed input straight through (NO removal). The do-nothing
                     floor: it scores exactly the shadowed-input shadow-region PSNR.
  * 'unet_nomask' -> a BLIND U-Net that sees only the 3-channel shadowed RGB and must both
                     LOCATE and correct the shadow from colour alone (the DeshadowNet-style
                     multi-context net WITHOUT the mask prior). It removes some shadow but,
                     not knowing exactly where/how-much, leaks into the lit region and over-
                     or under-corrects at the soft penumbra.
  * 'unet_mask'   -> the MASK-GUIDED U-Net: the soft shadow mask is concatenated as a 4th
                     input channel, so the net knows exactly WHERE and HOW MUCH to brighten
                     -- the SP+M-Net physically-parameterised recovery that fits the
                     multiplicative attenuation. The strong answer: highest shadow-region PSNR.

The DEFAULT below returns arch='copy' -> the do-nothing floor. A blind U-Net beats the
floor; the mask-guided U-Net gives clear further shadow-region PSNR headroom. A malformed /
crashing return falls back to arch='unet_mask'.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — design the deshadower backbone below
# ================================================================
def get_network_config():
    # Default: copy the shadowed input through (no removal) -> the do-nothing floor.
    return {"arch": "copy"}
# ================================================================
# END EDITABLE REGION
# ================================================================
