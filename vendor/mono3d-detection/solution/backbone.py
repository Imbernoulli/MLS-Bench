"""Monocular-3D HEAD-CAPACITY design surface (agent-editable) for mono3d-head-capacity.

A refinement block maps the shared 128-d encoder embedding to a refined 128-d embedding before
the fixed task heads. Its depth/width sets how much the representation can disentangle the
depth/pose factors — but too narrow a block bottlenecks the information.

You implement:

    def build_backbone(emb_dim) -> nn.Module:

The module maps emb [B, emb_dim] -> emb [B, emb_dim].

The DEFAULT below is the WEAK baseline: a SHALLOW/NARROW block that squeezes the 128-d embedding
through an 8-d bottleneck with NO residual, destroying most of the encoded information. A wider,
deeper RESIDUAL refinement (skip connections preserve information while adding capacity) is far
stronger. Everything else is fixed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# EDITABLE REGION — build_backbone(emb_dim) -> nn.Module (emb -> emb)
# ================================================================
def build_backbone(emb_dim):
    # WEAK DEFAULT: narrow 8-d bottleneck, NO residual -> hard information squeeze.
    class _Narrow(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(emb_dim, 8), nn.ReLU(), nn.Linear(8, emb_dim))

        def forward(self, x):
            return self.f(x)

    return _Narrow()
# ================================================================
# END EDITABLE REGION
# ================================================================
