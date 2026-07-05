"""Monocular-3D DIMENSION-PRIOR design surface (agent-editable) for mono3d-dimension-prior.

Predict the object's metric box dimensions (l, h, w). Dimensions have a strong, low-variance
per-class prior (a car is ~1.5 m tall). The METRIC HEIGHT you predict also feeds the fixed
geometry depth Z = f*H/h2d, so a wrong H hurts depth too.

You implement:

    def build_dims_head(emb_dim, log_mean, cls_dims) -> (head: nn.Module, decode: callable):

`head(emb)` maps the shared embedding [B, emb_dim] to a raw tensor. `decode(raw, ctx) -> dims
[B,3]` turns it into POSITIVE metric (l, h, w). `log_mean` [3] is log of the mean canonical
dims (a shape prior); `cls_dims` [3,3] the per-class canonical dims.

The DEFAULT below is the WEAK baseline: it regresses the metric dims DIRECTLY (softplus) with NO
prior. Anchoring a small residual on the log class-mean (exp(log_mean + 0.3*raw)) is far stronger
— accurate dims immediately, and a correct H tightens the geometry depth. Everything else (data,
splits, encoder, depth/orient heads, optimizer, epochs, seed, scoring) is fixed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# EDITABLE REGION — build the dims head + decode(raw, ctx) -> dims
# ================================================================
def build_dims_head(emb_dim, log_mean, cls_dims):
    # WEAK DEFAULT: regress metric dims DIRECTLY (softplus), ignoring the class-mean prior.
    head = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 3))

    def decode(raw, ctx):
        return F.softplus(raw) + 0.05

    return head, decode
# ================================================================
# END EDITABLE REGION
# ================================================================
