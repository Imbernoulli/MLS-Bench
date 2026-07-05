"""Monocular-3D UNCERTAINTY-WEIGHTING design surface (agent-editable) for mono3d-uncertainty-weighting.

The shared encoder is trained on three losses — depth, orientation, dimensions — that must be
balanced. The geometry depth Z=f*H/h2d depends on the predicted metric height H (dims head) and
the depth head's residual, so starving the depth/dims losses breaks the depth.

You implement:

    def build_task_weighting() -> (params: nn.Module|None, weight: callable):

`params` holds any learnable weighting parameters (register them so they train), or nn.Identity()
for a fixed scheme. `weight(losses: dict) -> scalar` combines the per-task losses; `losses` has
keys "depth", "orient", "dims".

The DEFAULT below is the WEAK baseline: a DEGENERATE fixed weighting that all but ignores the
depth and dimension losses (weight 0.001) -> untrained H / depth residual -> broken geometry
depth. A learned HOMOSCEDASTIC (Kendall) uncertainty weighting keeps every task supervised and
adaptively balances them. Everything else is fixed.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ================================================================
# EDITABLE REGION — build_task_weighting() -> (params, weight)
# ================================================================
def build_task_weighting():
    # WEAK DEFAULT: degenerate fixed weights that starve depth+dims -> broken geometry depth.
    def weight(losses):
        return 0.001 * losses["depth"] + losses["orient"] + 0.001 * losses["dims"]

    return nn.Identity(), weight
# ================================================================
# END EDITABLE REGION
# ================================================================
