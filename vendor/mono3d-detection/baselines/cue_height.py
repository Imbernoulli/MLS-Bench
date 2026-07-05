"""mono3d-depth-cue STRONG baseline: geometry depth from the 2D-box HEIGHT (Z = f*H/h2d).

Recover depth from the object's metric HEIGHT and the 2D-box pixel HEIGHT: Z = f*H/h2d. Vertical
extent is the RIGHT projective cue — an object's metric height is (near-)invariant to yaw, so the
2D height depends only on distance, making Z=f*H/h2d a clean inverse-depth relation. Lowest depth
error and best AP3D across all regimes. Reference: Deep3DBox / GS3D / GUPNet height-guided depth.
"""
import torch
import torch.nn as nn


def build_depth_head(emb_dim):
    head = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def decode(raw, ctx):
        H = ctx["pred_H"].reshape(-1)
        h2d = ctx["h2d"].reshape(-1).clamp(min=1.0)
        geom = ctx["focal"] * H / h2d
        return geom * torch.exp(0.1 * raw[:, 0].clamp(-3, 3))

    return head, decode
