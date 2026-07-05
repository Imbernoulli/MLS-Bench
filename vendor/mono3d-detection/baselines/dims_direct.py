"""mono3d-dimension-prior WEAK baseline: DIRECT dimension regression (no shape prior).

Regress the metric (l, h, w) DIRECTLY from the embedding with no statistical shape prior.
Object dimensions have a strong, low-variance per-class prior (a car is ~1.5 m tall); ignoring
it forces the head to learn absolute metric scale from scratch, so early/under-fit predictions
are far off and — critically — a WRONG height H feeds the geometry depth Z=f*H/h2d, coupling a
bad dims estimate into a bad depth. High dim error and (via H) worse AP3D. Reference: Deep3DBox
/ MonoDLE both regress a residual on the class-mean dims, not the raw metric size.
"""
import torch
import torch.nn as nn


def build_dims_head(emb_dim, log_mean, cls_dims):
    head = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 3))

    def decode(raw, ctx):
        # DIRECT positive dims via softplus, ignoring the class-mean prior entirely.
        return torch.nn.functional.softplus(raw) + 0.05

    return head, decode
