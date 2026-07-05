"""Monocular-3D PROJECTION design surface (agent-editable) for mono3d-projection.

Given the estimated depth Z, recover the 3D box CENTER's lateral/vertical position (x, y). The
object is generally OFF the camera axis (laterally offset, below the horizon), so its 3D center
must be back-projected from its 2D-box center through the pinhole camera model.

You implement:

    def build_backproject() -> callable:

`backproject(loc_z, box2d, cx, cy, focal) -> (x [B], y [B])` maps the predicted depth and the
amodal 2D box [B,4] to the 3D center's x, y. The pinhole inverse projection is
    x = (u - cx) * Z / f ,  y = (v - cy) * Z / f ,
where (u, v) is the 2D-box center and (cx, cy) the principal point.

The DEFAULT below is the WEAK baseline: it assumes every object is ON the camera axis (x=y=0),
ignoring the 2D position entirely — a large center error for off-axis objects that collapses the
3D IoU even with a perfect depth. The full pinhole back-projection is far stronger. Everything
else (data, splits, encoder, all heads, optimizer, epochs, seed, scoring) is fixed.
"""
from __future__ import annotations

import torch


# ================================================================
# EDITABLE REGION — build backproject(loc_z, box2d, cx, cy, focal) -> (x, y)
# ================================================================
def build_backproject():
    # WEAK DEFAULT: ON-AXIS assumption — place every object at x=0, y=0 (ignore 2D position).
    def backproject(loc_z, box2d, cx, cy, focal):
        z = loc_z.reshape(-1)
        return torch.zeros_like(z), torch.zeros_like(z)

    return backproject
# ================================================================
# END EDITABLE REGION
# ================================================================
