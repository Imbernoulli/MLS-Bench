"""mono3d-height-source WEAK baseline: geometry depth with a GLOBAL CONSTANT height.

Geometry depth Z = f*H0/h2d using a single GLOBAL CONSTANT height H0 = 1.5 m for EVERY object,
ignoring the per-object / per-class metric height. The scene mixes cars (~1.5 m), pedestrians
(~1.8 m) and cyclists (~1.7 m) with per-object jitter, so a constant H0 mis-scales depth for every
non-car object (Z is off by the true-H / H0 ratio) -> systematically biased depth, higher error
and lower AP3D. Reference: the height H in Z=f*H/h2d must be the object's OWN height; a global
constant is a known-bad simplification.
"""
import torch
import torch.nn as nn


def build_depth_head(emb_dim):
    head = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def decode(raw, ctx):
        h2d = ctx["h2d"].reshape(-1).clamp(min=1.0)
        H0 = torch.full_like(h2d, 1.5)                    # GLOBAL constant height for ALL classes
        geom = ctx["focal"] * H0 / h2d
        return geom * torch.exp(0.1 * raw[:, 0].clamp(-3, 3))

    return head, decode
