"""mono3d-height-source STRONG baseline: geometry depth with the PER-OBJECT predicted height.

Geometry depth Z = f*H/h2d using the PER-OBJECT metric height H from the dims head (which is
anchored on the class-mean prior, so it correctly distinguishes a ~1.5 m car from a ~1.8 m
pedestrian from a ~1.7 m cyclist). A correct H makes Z=f*H/h2d accurate for every class.
Reference: Deep3DBox / GUPNet use the estimated object height, not a global constant.
"""
import torch
import torch.nn as nn


def build_depth_head(emb_dim):
    head = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def decode(raw, ctx):
        H = ctx["pred_H"].reshape(-1)                     # per-object predicted metric height
        h2d = ctx["h2d"].reshape(-1).clamp(min=1.0)
        geom = ctx["focal"] * H / h2d
        return geom * torch.exp(0.1 * raw[:, 0].clamp(-3, 3))

    return head, decode
