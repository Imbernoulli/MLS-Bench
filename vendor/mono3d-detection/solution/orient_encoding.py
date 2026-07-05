"""Monocular-3D ORIENTATION-ENCODING design surface (agent-editable) for
mono3d-orientation-encoding.

Recover the object's yaw (rotation about the vertical axis) from a single image region. The
difficulty is the 2*pi wrap-around: yaw lives on a circle, so a plain scalar L1/L2 regressor
is discontinuous at +-pi and unstable — worst for broadside objects (|yaw| near pi).

You implement:

    def build_orient_head(emb_dim) -> (head: nn.Module, decode: callable, loss: callable):

`head(emb)` maps the shared embedding [B, emb_dim] to a raw tensor. `decode(raw) -> yaw [B]`
turns it into a yaw angle (radians). `loss(raw, yaw_gt) -> scalar` is the training loss.

The DEFAULT below is the WEAK baseline: regress yaw as ONE scalar with a plain smooth-L1 to
the GT angle. It has no notion of the circle -> unstable at the +-pi wrap, high angular error.
Stronger encodings: regress (cos, sin) and take atan2 (removes the wrap discontinuity), or the
Deep3DBox MULTIBIN scheme (classify which angular bin + regress a residual within the bin).
Everything else (data, splits, encoder, dims head, geometry-depth, optimizer, epochs, seed,
scoring) is fixed; the only degree of freedom is this orientation parameterization.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# EDITABLE REGION — build orient head + decode(raw)->yaw + loss(raw,yaw_gt)
# ================================================================
def build_orient_head(emb_dim):
    # WEAK DEFAULT: regress yaw as a single scalar with plain smooth-L1 to the GT angle.
    # No circular structure -> unstable at the +-pi wrap, worst for broadside objects.
    head = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def decode(raw):
        return raw[:, 0]

    def loss(raw, yaw_gt):
        return F.smooth_l1_loss(raw[:, 0], yaw_gt)

    return head, decode, loss
# ================================================================
# END EDITABLE REGION
# ================================================================
