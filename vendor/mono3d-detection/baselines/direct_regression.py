"""Reference (WEAK): direct metric-depth regression for mono3d-depth-parameterization."""
import torch.nn as nn
import torch.nn.functional as F


def build_depth_head(emb_dim):
    head = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def decode(raw, ctx):
        return F.softplus(raw[:, 0]) + 1.0

    return head, decode
