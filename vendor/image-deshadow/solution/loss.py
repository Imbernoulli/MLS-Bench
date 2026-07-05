"""Agent-editable surface: RECONSTRUCTION LOSS of the mask-guided deshadower.

A FIXED mask-guided residual deshadower (same U-Net width/depth, optimiser, data, iters,
seed, eval split) removes a cast shadow (physics-based linear illumination model  I = a*J,
SP+M-Net, Le et al. ICCV 2019). A shadow-up-weighted L1 term is ALWAYS present; you design
which extra, physically-grounded terms are added:

    def get_loss_config():
        # {'ssim': bool, 'color': bool, 'comp': bool}
        return {'ssim': True, 'color': True, 'comp': True}

  * ssim  -> add a structural SSIM term (sharper penumbra edge, less over-smoothing).
  * color -> add a CHROMA-consistency term (mean-subtracted RGB): penalises the colour
             cast a naive brightening leaves in the recovered umbra.
  * comp  -> add a re-shadow COMPOSITION-consistency term: require the recovery to agree
             with the input where the shadow is weak (do not corrupt the already-lit
             pixels) -- the SP+M-Net decomposition-consistency idea.

The DEFAULT below returns all three OFF (plain shadow-weighted L1 only). Adding the SSIM +
color + composition terms sharpens the recovered shadow region -> higher shadow-region PSNR.
A malformed / crashing return falls back to all terms OFF (weak plain-L1).
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — design the loss composition below
# ================================================================
def get_loss_config():
    # Default: plain shadow-weighted L1 only (no SSIM / color / composition terms).
    return {"ssim": False, "color": False, "comp": False}
# ================================================================
# END EDITABLE REGION
# ================================================================
