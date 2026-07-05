"""mono3d-3d-box-loss STRONG baseline: DECOUPLED / disentangled per-component loss.

Score each 3D-box component in its OWN well-scaled space: log-depth (scale-invariant, so near
and far objects contribute comparably), log-dims (relative dimension error), and an ANGULAR yaw
loss (sin of the wrapped residual, continuous at +-pi). Decoupling puts every term on a
comparable footing and removes the metre-scale depth domination and the yaw discontinuity, so
all three quantities train well together. Reference: MonoDLE / "Disentangling Monocular 3D
Object Detection" — disentangled per-group losses beat the coupled raw-unit loss.
"""
import torch
import torch.nn.functional as F


def build_loss3d(emb_dim):
    def loss(pred, gt):
        ld = F.smooth_l1_loss(torch.log(pred["Z"].clamp(min=0.5)), torch.log(gt["Z"]))
        ldim = F.smooth_l1_loss(torch.log(pred["dims"].clamp(min=0.05)), torch.log(gt["dims"]))
        dyaw = torch.atan2(torch.sin(pred["yaw"] - gt["yaw"]), torch.cos(pred["yaw"] - gt["yaw"]))
        lyaw = (dyaw ** 2).mean()                            # angular, continuous at +-pi
        return ld + 0.5 * ldim + 0.5 * lyaw

    return loss
