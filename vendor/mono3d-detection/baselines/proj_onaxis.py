"""mono3d-projection WEAK baseline: ON-AXIS assumption (object centered, x=y=0).

Ignore the image position and place every object ON the camera axis: x = 0, y = 0, keeping only
the estimated depth Z. But objects are laterally offset (x up to ~0.35*Z) and below the horizon
(y up to ~0.25*Z); assuming they are centered puts the 3D box in the wrong world location for
every off-axis object, so even a perfect depth gives a large center error and the BEV/3D IoU
collapses. This is the degenerate that ignores the pinhole back-projection. Reference: the lateral
/ vertical world position MUST be recovered from the 2D center via the pinhole model.
"""
import torch


def build_backproject():
    def backproject(loc_z, box2d, cx, cy, focal):
        z = loc_z.reshape(-1)
        x = torch.zeros_like(z)          # ON-AXIS: ignore the 2D-box horizontal position
        y = torch.zeros_like(z)          # ON-AXIS: ignore the 2D-box vertical position
        return x, y

    return backproject
