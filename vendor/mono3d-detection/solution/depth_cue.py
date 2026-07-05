"""Monocular-3D DEPTH-CUE design surface (agent-editable) for mono3d-depth-cue.

Recover metric depth Z from the projective geometry, choosing WHICH 2D-box extent drives it.
An object of metric size S projecting to a 2D extent p px at depth Z gives Z = f*S/p. Two cues
are available: the box HEIGHT (Z = f*H/h2d) and the box WIDTH (Z = f*W/w2d).

You implement:

    def build_depth_head(emb_dim) -> (head: nn.Module, decode: callable):

`decode(raw, ctx) -> Z [B]` (POSITIVE metric depth) may use:
    ctx["focal"]  scalar focal length f (px)
    ctx["h2d"]    [B] 2D-box pixel HEIGHT      ctx["pred_H"] [B] predicted metric height H
    ctx["w2d"]    [B] 2D-box pixel WIDTH       ctx["pred_W"] [B] predicted metric width W

The DEFAULT below is the WEAK baseline: it uses the box WIDTH (Z = f*W/w2d). But the amodal box
WIDTH is confounded by yaw (under full-circle rotation the width sweeps between the object's
length and width), so it is a POOR depth cue. The object HEIGHT is (near-)yaw-invariant, so
Z = f*H/h2d is far more accurate. Everything else is fixed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# EDITABLE REGION — build the depth head + decode(raw, ctx) -> Z
# ================================================================
def build_depth_head(emb_dim):
    # WEAK DEFAULT: geometry depth from the (yaw-confounded) 2D-box WIDTH: Z = f*W / w2d.
    head = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def decode(raw, ctx):
        W = ctx["pred_W"].reshape(-1)
        w2d = ctx["w2d"].reshape(-1).clamp(min=1.0)
        geom = ctx["focal"] * W / w2d
        return geom * torch.exp(0.1 * raw[:, 0].clamp(-3, 3))

    return head, decode
# ================================================================
# END EDITABLE REGION
# ================================================================
