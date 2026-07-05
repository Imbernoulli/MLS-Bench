"""Agent-editable surface: NORMALIZATION in the mask-guided deshadower.

A FIXED mask-guided residual deshadower (same U-Net width/depth, loss, optimiser, data,
iters, seed, eval split) removes a cast shadow (SP+M-Net linear illumination model, Le et al.
ICCV 2019). You design ONLY the normalisation layer inside the residual blocks:

    def get_norm_config():
        # {'norm': 'none' | 'bn' | 'in'}
        return {'norm': 'in'}

  * norm='none' -> no normalisation. WEAK.
  * norm='bn'   -> batch normalisation.
  * norm='in'   -> INSTANCE normalisation: normalises per-image feature statistics, which
                   removes the global illumination bias a cast shadow introduces so the net
                   focuses on the LOCAL relighting -> higher shadow-region PSNR.

The DEFAULT below returns norm='none'. Instance norm typically stabilises the illumination
recovery and raises shadow-region PSNR. A malformed / crashing return falls back to
norm='none' (weak).
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — design the normalization config below
# ================================================================
def get_norm_config():
    # Default: no normalization.
    return {"norm": "none"}
# ================================================================
# END EDITABLE REGION
# ================================================================
