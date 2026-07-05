"""mono3d-projection STRONG baseline: full PINHOLE back-projection of the 2D-box center.

Recover the 3D center (x, y) by back-projecting the amodal 2D-box center through the pinhole
model at the predicted depth Z:  x = (u - cx) * Z / f ,  y = (v - cy) * Z / f. This places the
object at the correct lateral/vertical world position for its image location — essential for
off-axis objects (which are most of them). Reference: standard pinhole inverse projection used by
every monocular-3D detector to lift the 2D center to 3D.
"""
import torch


def build_backproject():
    def backproject(loc_z, box2d, cx, cy, focal):
        u = 0.5 * (box2d[:, 0] + box2d[:, 2])
        v = 0.5 * (box2d[:, 1] + box2d[:, 3])
        z = loc_z.reshape(-1)
        x = (u - cx) * z / focal
        y = (v - cy) * z / focal
        return x, y

    return backproject
