import torch
import torch.nn as nn

# =====================================================================
# EDITABLE: implement run_attack below
# =====================================================================
def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    pixels: int,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    """
    Sparse L0 adversarial attack.
    images: (N, C, H, W) in [0, 1] on device. labels: (N,) on device.
    pixels: max number of modified spatial pixels (H, W) per sample.
    n_classes: 10 for CIFAR-10, 100 for CIFAR-100.
    Returns adv_images satisfying an L0 pixel budget validated by evaluator.
    """
    _ = (model, labels, pixels, device, n_classes)
    return images.clone()

# =====================================================================
# END EDITABLE REGION
# =====================================================================
