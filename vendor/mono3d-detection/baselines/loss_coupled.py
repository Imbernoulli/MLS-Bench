"""mono3d-3d-box-loss WEAK baseline: COUPLED raw-metric L2 on the 7-DoF vector.

Combine the depth/dims/yaw errors as a single L2 on the raw concatenated 7-DoF box vector
[Z, l, h, w, x, y, yaw] in their native units. Depth spans 6-40 m while dims are ~1-4 m and yaw
is in radians, so the metre-scale depth term (and far objects within it) DOMINATES the gradient;
dims/yaw are under-trained and the yaw wrap-around at +-pi is not handled -> the coupled loss
trains an unbalanced box. Reference: MonoDLE / "Disentangling Monocular 3D Object Detection"
show coupling the 3D-box terms in raw units is inferior to a decoupled/disentangled loss.
"""
import torch
import torch.nn.functional as F


def build_loss3d(emb_dim):
    def loss(pred, gt):
        # raw-unit coupled L2: metre-scale depth dominates, yaw wrap unhandled.
        z = (pred["Z"] - gt["Z"]) ** 2
        d = ((pred["dims"] - gt["dims"]) ** 2).sum(dim=1)
        y = (pred["yaw"] - gt["yaw"]) ** 2                   # NO circular handling
        return (z + d + y).mean()

    return loss
