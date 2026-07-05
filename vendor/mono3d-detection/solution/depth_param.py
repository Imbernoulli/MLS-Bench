"""Monocular-3D DEPTH-PARAMETERIZATION design surface (agent-editable) for
mono3d-depth-parameterization.

Recover the single hardest quantity in monocular 3D detection: the object's METRIC depth Z
(the camera-frame distance of the box center). Projection destroys absolute scale, so Z is
genuinely ill-posed from appearance alone — the winning cue is projective GEOMETRY: an object
of true metric height H projects to a 2D box of pixel height h2d ~ f * H / Z, hence
Z ~ f * H / h2d (Deep3DBox / GS3D / height-guided depth).

You implement:

    def build_depth_head(emb_dim) -> (head: nn.Module, decode: callable):

`head(emb)` maps the shared RegionEncoder embedding [B, emb_dim] to a raw tensor of your
choosing. `decode(raw, ctx) -> Z [B]` turns that raw output into a POSITIVE metric depth,
and may use the projective context `ctx`:
    ctx["focal"]  scalar pinhole focal length f (pixels)
    ctx["h2d"]    [B]  pixel HEIGHT of the amodal 2D box (the inverse-depth cue)
    ctx["pred_H"] [B]  predicted metric object height H (from the fixed dims head)
    ctx["box2d"]  [B,4] amodal 2D box, ctx["cx"], ctx["cy"] principal point

The DEFAULT below is the WEAK baseline: it regresses Z DIRECTLY as an unbounded scalar and
ignores the geometry. Direct depth regression is dominated by far objects and generalizes
poorly across distance — low AP3D, especially at range. A projective decode
(Z = f * pred_H / h2d, optionally * a small learned residual) is far stronger and its edge
GROWS with distance. Everything else (data, splits, encoder, dims head, optimizer, epochs,
seed, scoring) is fixed; the only degree of freedom is this depth parameterization.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# EDITABLE REGION — build the depth head + decode(raw, ctx) -> Z
# ================================================================
def build_depth_head(emb_dim):
    # WEAK DEFAULT: regress metric depth DIRECTLY (softplus for positivity), ignoring the
    # projective geometry. Far objects dominate; poor AP3D at range.
    head = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def decode(raw, ctx):
        # raw[:, 0] -> metric depth via softplus; NO use of focal / h2d / pred_H.
        return F.softplus(raw[:, 0]) + 1.0

    return head, decode
# ================================================================
# END EDITABLE REGION
# ================================================================
