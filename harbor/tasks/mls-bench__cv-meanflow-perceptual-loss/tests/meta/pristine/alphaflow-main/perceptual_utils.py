
# Perceptual Loss Utilities for Flow Matching
# Added by MLS-Bench pre_edit

import torch
import torch.nn.functional as F

def compute_gradient_loss(x_pred, x_target):
    """Compute gradient loss for edge sharpness.

    Args:
        x_pred: [B, C, H, W] predicted images
        x_target: [B, C, H, W] target images

    Returns:
        scalar loss
    """
    # Sobel filters for x and y gradients
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=x_pred.dtype, device=x_pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=x_pred.dtype, device=x_pred.device).view(1, 1, 3, 3)

    # Compute gradients for each channel
    grad_loss = 0
    for c in range(x_pred.shape[1]):
        pred_c = x_pred[:, c:c+1]
        target_c = x_target[:, c:c+1]

        pred_grad_x = F.conv2d(pred_c, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_c, sobel_y, padding=1)
        target_grad_x = F.conv2d(target_c, sobel_x, padding=1)
        target_grad_y = F.conv2d(target_c, sobel_y, padding=1)

        grad_loss += F.l1_loss(pred_grad_x, target_grad_x)
        grad_loss += F.l1_loss(pred_grad_y, target_grad_y)

    return grad_loss / x_pred.shape[1]


def compute_multiscale_loss(x_pred, x_target, scales=[0.5, 0.25]):
    """Compute multi-scale MSE loss.

    Args:
        x_pred: [B, C, H, W] predicted images
        x_target: [B, C, H, W] target images
        scales: list of downsampling scales

    Returns:
        scalar loss
    """
    loss = 0
    for scale in scales:
        x_pred_scaled = F.interpolate(x_pred, scale_factor=scale, mode='bilinear', align_corners=False)
        x_target_scaled = F.interpolate(x_target, scale_factor=scale, mode='bilinear', align_corners=False)
        loss += F.mse_loss(x_pred_scaled, x_target_scaled)
    return loss / len(scales)
