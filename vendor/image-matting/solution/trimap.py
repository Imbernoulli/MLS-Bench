"""Agent-editable surface: the TRIMAP ENCODING / CONDITIONING.

Return a function
    encode_trimap(trimap) -> tensor (B, K, H, W)
that turns the trimap (B,H,W in {0,0.5,1}) into K feature planes that are
concatenated to the RGB image (3 channels) and fed to the FIXED U-Net matting
encoder. The encoder, decoder, loss (fixed alpha-L1 + composition), optimiser,
iterations, seed and eval are FIXED; only the trimap encoding changes. Scored by
SAD (LOWER is better) in the trimap UNKNOWN band on a held-out val split.

    def encode_trimap(trimap):
        import torch
        fg  = (trimap > 0.75).float()      # definite foreground
        bg  = (trimap < 0.25).float()      # definite background
        unk = ((trimap >= 0.25) & (trimap <= 0.75)).float()   # unknown band
        return torch.stack([fg, unk, bg], dim=1)   # (B,3,H,W) one-hot trimap

Trimap conditioning RESOLVES the foreground/background ambiguity in the unknown
region: without it the net only sees pixel colour (ambiguous where fg/bg colours
overlap) and cannot know which side of the matte a pixel belongs to, so it defaults
to a smeared alpha -> high SAD. Feeding the trimap (as the raw channel, Deep Image
Matting Xu et al. 2017, or as a 3-plane one-hot fg/unk/bg encoding) tells the net
exactly where the solved regions are, so it only has to interpolate the soft matte
across the KNOWN boundary -> much lower SAD.

The DEFAULT below is a deliberately UNINFORMATIVE encoding: an all-zeros plane
(the trimap signal is thrown away), so the net is effectively trimap-blind and must
guess the matte from colour alone -> high SAD. Returning the real trimap information
(raw channel or one-hot planes) recovers accuracy with clear headroom. A malformed
/ crashing return falls back to the raw trimap channel.
"""
from __future__ import annotations

import torch


# ================================================================
# EDITABLE REGION — design the trimap encoding below
# ================================================================
def encode_trimap(trimap):
    # Default: an all-zeros plane -> throws away the trimap. The net is trimap-blind
    # and must infer the matte from RGB colour alone -> high SAD.
    return torch.zeros_like(trimap).unsqueeze(1)   # (B,1,H,W) all zeros
# ================================================================
# END EDITABLE REGION
# ================================================================
