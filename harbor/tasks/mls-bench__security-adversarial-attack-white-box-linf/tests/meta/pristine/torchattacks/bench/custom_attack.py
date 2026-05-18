import torch
import torch.nn as nn

# =====================================================================
# EDITABLE: implement run_attack below
# =====================================================================
def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    """
    White-box L_inf adversarial attack.
    images: (N, C, H, W) in [0, 1] on device. labels: (N,) on device.
    n_classes: 10 for CIFAR-10, 100 for CIFAR-100.
    Returns adv_images satisfying (adv_images - images).abs().max() <= eps.
    """
    _ = (model, labels, eps, device, n_classes)
    return images.clone()

# =====================================================================
# END EDITABLE REGION
# =====================================================================
