"""Reference (STRONG/SOTA): geometry depth Z = f*H/h2d * exp(0.1*residual) (Deep3DBox+refine)."""
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
