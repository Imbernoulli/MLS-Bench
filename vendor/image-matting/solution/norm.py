"""Agent-editable surface: the NORMALISATION layer (norm).

Return a callable make_norm(num_ch) -> torch.nn.Module that produces the
normalisation layer applied after each conv (inside every encoder/decoder block).
Everything else (data, encoder/decoder structure, trimap, loss, optimiser,
iterations, seed, eval) is FIXED; only the norm layer changes. Scored by SAD (LOWER
is better) in the trimap UNKNOWN band, gmean over three trimap-width settings.

Normalisation stabilises and speeds up training of the matting net. On this
synthetic composite data (recurring low-frequency fg/bg statistics across images):
    identity / no-norm  <  InstanceNorm (per-image)  <  BatchNorm (cross-image
    statistics, best on this recurring data = SOTA here).
BatchNorm's cross-image statistics are informative because the composite statistics
recur across the fixed synthetic set, so it converges to a lower SAD in the short
fine-tune.

The DEFAULT below is a deliberately weak IDENTITY (no normalisation): training is
slower / less stable in the short fine-tune -> higher SAD. Redesign make_norm() to
return nn.BatchNorm2d(num_ch) (or InstanceNorm) with clear headroom. A malformed /
crashing norm falls back to BatchNorm.
"""
from __future__ import annotations

import torch.nn as nn


# ================================================================
# EDITABLE REGION — design the normalisation layer below
# ================================================================
def make_norm(num_ch):
    # Default: IDENTITY (no normalisation). Without normalisation the short fine-tune
    # is slower / less stable -> higher SAD. BatchNorm (cross-image stats, informative
    # on this recurring synthetic data) converges to a much lower SAD.
    return nn.Identity()
# ================================================================
# END EDITABLE REGION
# ================================================================
