"""Agent-editable surface: the BOTTLENECK ATTENTION / CONTEXT block (attention).

Return a torch.nn.Module via build_attention(ch) whose forward maps a bottleneck
feature (B, ch, H, W) to the SAME shape (B, ch, H, W). It is inserted at the
stride-8 encoder bottleneck. Everything else (data, encoder, decoder, trimap, loss,
optimiser, iterations, seed, eval) is FIXED; only this block changes. Scored by SAD
(LOWER is better) in the trimap UNKNOWN band, gmean over three trimap-width settings.

Matting needs to aggregate CONTEXT so the unknown band knows which side of the matte
it belongs to. A global-average-pool block collapses all spatial context (worst); a
local 3x3 conv aggregates only a small neighbourhood; a non-local SELF-ATTENTION /
guided-context block aggregates GLOBAL context and is best (cf. contextual attention,
Yu et al. 2018). Order:
    global-pool  <  local-conv  <  self-attention (SOTA).

The DEFAULT below is a deliberately weak GLOBAL-AVERAGE-POOL block: it replaces every
spatial location with the channel-wise global mean (broadcast back), destroying all
spatial context in the bottleneck -> high SAD. Redesign build_attention() as a
non-local self-attention block (or at least a local conv) that preserves and
aggregates spatial context, with clear headroom. A malformed / crashing / wrong-shape
module falls back to identity.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# EDITABLE REGION — design the bottleneck attention/context block below
# ================================================================
def build_attention(ch):
    # Default: GLOBAL-AVERAGE-POOL block. Collapse every spatial location to the
    # channel-wise global mean (broadcast back) -> destroys all spatial context in
    # the bottleneck -> the unknown band cannot localise the matte -> high SAD.
    class GAPBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(ch, ch, 1)

        def forward(self, x):
            g = x.mean(dim=(-2, -1), keepdim=True)     # (B,ch,1,1) global mean
            g = self.proj(g)
            return g.expand_as(x)                       # broadcast -> no spatial info
    return GAPBlock()
# ================================================================
# END EDITABLE REGION
# ================================================================
