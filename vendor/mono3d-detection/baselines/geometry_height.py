"""Reference (MEDIUM): analytic geometry depth Z = f*H/h2d (no residual)."""
import torch.nn as nn


def build_depth_head(emb_dim):
    head = nn.Sequential(nn.Linear(emb_dim, 16), nn.ReLU(), nn.Linear(16, 1))

    def decode(raw, ctx):
        H = ctx["pred_H"].reshape(-1)
        h2d = ctx["h2d"].reshape(-1).clamp(min=1.0)
        return ctx["focal"] * H / h2d

    return head, decode
