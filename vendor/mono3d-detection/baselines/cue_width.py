"""mono3d-depth-cue WEAK baseline: geometry depth from the 2D-box WIDTH (Z = f*W/w2d).

Recover depth from the object's metric WIDTH and the 2D-box pixel WIDTH: Z = f*W/w2d. But the
2D-box WIDTH is a BAD depth cue: under a full-circle yaw the amodal box width sweeps between the
object's length (l) and width (w) depending on heading, so w2d is NOT a fixed function of a single
metric extent at a given distance — the width cue is confounded by orientation. Using it gives a
systematically wrong, yaw-dependent depth -> higher depth error and lower AP3D than the height
cue. Reference: monocular-3D depth uses the (yaw-invariant) HEIGHT cue, not width.
"""
import torch
import torch.nn as nn


def build_depth_head(emb_dim):
    head = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def decode(raw, ctx):
        W = ctx["pred_W"].reshape(-1)                    # predicted metric width
        w2d = ctx["w2d"].reshape(-1).clamp(min=1.0)       # 2D box pixel WIDTH (yaw-confounded)
        geom = ctx["focal"] * W / w2d
        return geom * torch.exp(0.1 * raw[:, 0].clamp(-3, 3))

    return head, decode
