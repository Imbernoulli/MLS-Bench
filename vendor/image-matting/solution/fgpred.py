"""Agent-editable surface: the OUTPUT HEAD (fgpred) — alpha-only vs joint alpha+FG.

Return a callable build_head(cin) -> torch.nn.Module whose forward takes the last
decoder feature (B, cin, H, W) and returns per-pixel LOGITS (B, K, H, W). Channel 0
is ALWAYS the alpha logit (sigmoid -> the scored alpha). If K >= 4, channels 1:4 are
supervised by the harness as a FOREGROUND-COLOUR prediction (an auxiliary task): the
harness adds a foreground L1 loss on sigmoid(channels 1:4) vs the GT foreground in
the unknown band. Everything else is FIXED; only the head changes. Scored by SAD
(LOWER is better) on the ALPHA channel in the trimap UNKNOWN band, gmean over three
trimap-width settings.

Jointly predicting the FOREGROUND colour alongside alpha is a matting-standard
regulariser: the shared decoder features must explain both the matte and the
foreground appearance, which sharpens the matte (Context-Aware Matting, Hou & Liu
2019; background-matting joint prediction). Order:
    alpha-only 1-ch head  <  joint alpha + foreground-colour head (aux FG
    supervision = SOTA).

The DEFAULT below is a deliberately weak ALPHA-ONLY head (K=1): no auxiliary FG
supervision -> the decoder features are optimised for alpha alone and are less
constrained -> slightly higher SAD. Redesign build_head() to emit K=4 channels
(alpha logit + 3 foreground-colour logits) so the harness adds the aux FG loss, with
clear headroom. A malformed / crashing head falls back to the default 1-ch head.
"""
from __future__ import annotations

import torch.nn as nn


# ================================================================
# EDITABLE REGION — design the output head below
# ================================================================
def build_head(cin):
    # Default: ALPHA-ONLY head (1 output channel). No auxiliary foreground-colour
    # prediction -> the decoder features are constrained only by the alpha loss ->
    # slightly higher SAD. A joint head (4 channels: alpha + FG-rgb) adds an aux FG
    # supervision that regularises the shared features and lowers SAD.
    return nn.Conv2d(cin, 1, 1)
# ================================================================
# END EDITABLE REGION
# ================================================================
