"""Agent-editable surface: the ENCODER-DECODER SKIP FUSION (skip).

Return a function
    fuse(dec_up, skip) -> fused tensor (B, C_dec + C_skip, H, W)
that fuses an upsampled decoder feature `dec_up` (B,C_dec,H,W) with the matching
encoder skip feature `skip` (B,C_skip,H,W). The result is passed to the next decoder
conv, which expects `C_dec + C_skip` channels (the default concat width). Everything
else (data, encoder, decoder convs, trimap, loss, optimiser, iterations, seed, eval)
is FIXED; only the fusion changes. Scored by SAD (LOWER is better) in the trimap
UNKNOWN band, gmean over three trimap-width settings.

Skip connections (U-Net) inject the encoder's high-resolution, low-level features
into the decoder so the matte keeps sharp boundaries. Dropping the skip loses that
detail; a down-weighted partial skip recovers some of it; the full-strength concat
skip (standard U-Net fusion) recovers the most. Order:
    drop-skip  <  half-strength skip  <  full concat skip (SOTA).

The DEFAULT below is a deliberately weak DROP-SKIP: it discards the encoder skip and
duplicates the decoder feature to fill the expected width, so no encoder detail
reaches the decoder -> a blurry matte -> high SAD. Redesign fuse() to concatenate the
real encoder skip (full U-Net fusion) with clear headroom. A malformed / crashing
fuse falls back to the full concat skip.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ================================================================
# EDITABLE REGION — design the skip fusion below
# ================================================================
def fuse(dec_up, skip):
    # Default: DROP the encoder skip. Fill the expected concat width by duplicating
    # the decoder feature (skip*0) so no encoder high-res detail reaches the decoder
    # -> blurry matte, high SAD. Full concat skip recovers the detail.
    return torch.cat([dec_up, torch.zeros_like(skip)], 1)
# ================================================================
# END EDITABLE REGION
# ================================================================
