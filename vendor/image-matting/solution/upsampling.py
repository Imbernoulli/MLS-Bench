"""Agent-editable surface: the DECODER UPSAMPLING operator (upsampling).

Return a callable build_upsampler(cin) -> torch.nn.Module whose forward takes a
decoder feature (B, cin, H, W) and returns (B, cin, 2H, 2W) — an UPSAMPLED feature
with the SAME channel count (the harness resizes to the exact skip size if needed).
It is used at every decoder up-step. Everything else is FIXED; only the upsampler
changes. Scored by SAD (LOWER is better) in the trimap UNKNOWN band, gmean over three
trimap-width settings.

The upsampling operator determines how much soft-edge detail survives the decoder:
    nearest-neighbour (blocky, aliases the soft matte edges)  <  bilinear (smooth)
      <  a learned / guided upsample (transposed conv or bilinear + refine conv,
         sharpest soft matte = SOTA).

The DEFAULT below is a deliberately weak NEAREST-NEIGHBOUR upsample: it replicates
pixels, producing a blocky matte whose soft transition is aliased -> higher SAD and
gradient error. Redesign build_upsampler() as a learned upsampler (e.g. bilinear
upsample followed by a 3x3 refine conv, or a transposed conv) that reconstructs a
smooth, sharp soft edge, with clear headroom. A malformed / crashing / wrong-shape
module falls back to bilinear.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# EDITABLE REGION — design the decoder upsampling operator below
# ================================================================
def build_upsampler(cin):
    # Default: NEAREST-NEIGHBOUR upsample (x2). Replicates pixels -> blocky, aliased
    # soft edges -> higher SAD / gradient error. A learned upsample (bilinear + refine
    # conv, or transposed conv) reconstructs a smooth sharp soft edge.
    class NearestUp(nn.Module):
        def forward(self, x):
            return F.interpolate(x, scale_factor=2, mode="nearest")
    return NearestUp()
# ================================================================
# END EDITABLE REGION
# ================================================================
