"""Monocular-3D HEIGHT-SOURCE design surface (agent-editable) for mono3d-height-source.

Recover metric depth from the projective height relation Z = f*H/h2d. The QUESTION is where the
metric height H comes from. The scene mixes classes with different heights (car ~1.5 m,
pedestrian ~1.8 m, cyclist ~1.7 m) plus per-object jitter.

You implement:

    def build_depth_head(emb_dim) -> (head: nn.Module, decode: callable):

`decode(raw, ctx) -> Z [B]` may use:
    ctx["focal"]  scalar focal f (px)   ctx["h2d"] [B] 2D-box pixel height
    ctx["pred_H"] [B] the PER-OBJECT predicted metric height (from the fixed dims head)

The DEFAULT below is the WEAK baseline: it uses a single GLOBAL CONSTANT height H0 = 1.5 m for
EVERY object, so depth is mis-scaled for every non-car class. Using the PER-OBJECT ctx["pred_H"]
(which is anchored on the class prior) makes Z=f*H/h2d accurate for all classes. Everything else
is fixed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# EDITABLE REGION — build the depth head + decode(raw, ctx) -> Z
# ================================================================
def build_depth_head(emb_dim):
    # WEAK DEFAULT: geometry depth with a GLOBAL CONSTANT height H0=1.5 for ALL classes.
    head = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def decode(raw, ctx):
        h2d = ctx["h2d"].reshape(-1).clamp(min=1.0)
        H0 = torch.full_like(h2d, 1.5)
        geom = ctx["focal"] * H0 / h2d
        return geom * torch.exp(0.1 * raw[:, 0].clamp(-3, 3))

    return head, decode
# ================================================================
# END EDITABLE REGION
# ================================================================
