"""Agent-editable surface: GLOBAL RESIDUAL learning for the deblur net.

A FIXED compact residual encoder-decoder deblur net is trained to restore sharp
images from motion-blurred ones. You design ONLY whether the network predicts a
GLOBAL RESIDUAL correction added to the blurry input, or predicts the full sharp
image directly.

    def get_residual_config():
        # return {'global_residual': True|False}
        return {'global_residual': True}

With global residual ON, the output is  sharp_hat = blurry + net(blurry)  -- the net
only has to learn the (small, high-frequency) deblur CORRECTION, which is much easier
to optimise and yields sharper restorations. This long identity skip is used by every
strong deblur net (DeepDeblur, Nah et al. CVPR 2017; SRN-DeblurNet, Tao et al. CVPR
2018; MPRNet, Zamir et al. CVPR 2021).

The DEFAULT below returns global_residual=False -> the net must regress the FULL image
from scratch, a harder optimisation that (at this budget) leaves the output blurrier
and the deblur PSNR lower. Turning global residual ON gives clear PSNR headroom above
the direct-prediction floor (and both must clearly BEAT the blurry-input identity
floor, reported as blurry_psnr). A malformed / crashing return falls back to
global_residual=True.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — design the residual configuration below
# ================================================================
def get_residual_config():
    # Default: predict the FULL image directly (no global residual) -> harder, blurrier.
    return {"global_residual": False}
# ================================================================
# END EDITABLE REGION
# ================================================================
