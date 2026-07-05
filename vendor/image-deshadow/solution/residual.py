"""Agent-editable surface: RESIDUAL LEARNING in the mask-guided deshadower.

A FIXED mask-guided deshadower (same U-Net width/depth, loss, optimiser, data, iters, seed,
eval split) removes a cast shadow (SP+M-Net linear illumination model  I = a*J, Le et al.
ICCV 2019). You design ONLY whether the net regresses the clean image directly or predicts a
RESIDUAL correction:

    def get_residual_config():
        # {'mode': 'direct' | 'residual'}
        return {'mode': 'residual'}

  * mode='direct'   -> the net regresses the CLEAN image DIRECTLY (clean = net(.)). It must
                       reconstruct the whole scene including the already-lit region from
                       scratch -> over-smooths, wastes capacity -> lower shadow-region PSNR.
  * mode='residual' -> RESIDUAL learning: clean = shadowed + net(.), the net predicts only
                       the shadow CORRECTION. For the near-multiplicative degradation this is
                       a far easier target (the lit region needs ~zero correction) -> higher
                       shadow-region PSNR. The strong answer.

The DEFAULT below returns mode='direct'. Residual learning of the correction typically raises
shadow-region PSNR. A malformed / crashing return falls back to mode='direct' (weak).
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — design the residual-learning config below
# ================================================================
def get_residual_config():
    # Default: direct clean-image regression (no residual learning).
    return {"mode": "direct"}
# ================================================================
# END EDITABLE REGION
# ================================================================
