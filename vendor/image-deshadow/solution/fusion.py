"""Agent-editable surface: FEATURE FUSION in the mask-guided deshadower.

A FIXED mask-guided residual deshadower (same U-Net width/depth, loss, optimiser, data,
iters, seed, eval split) removes a cast shadow (SP+M-Net linear illumination model, Le et al.
ICCV 2019). You design ONLY how decoder features are aggregated before the output conv:

    def get_fusion_config():
        # {'fusion': True | False}
        return {'fusion': True}

  * fusion=False -> use only the LAST decoder block's features. WEAK.
  * fusion=True  -> DENSE multi-level FEATURE FUSION: concatenate features from every
                    decoder level and fuse them with a 1x1 conv (DenseNet / RDN-style), so
                    both coarse global-illumination and fine penumbra-edge features feed the
                    output -> higher shadow-region PSNR.

The DEFAULT below returns fusion=False. Dense multi-level fusion typically raises shadow-
region PSNR. A malformed / crashing return falls back to fusion=False (weak).
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — design the fusion config below
# ================================================================
def get_fusion_config():
    # Default: last-block features only (no dense fusion).
    return {"fusion": False}
# ================================================================
# END EDITABLE REGION
# ================================================================
