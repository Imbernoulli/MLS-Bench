"""Custom regularizer for 3D Gaussian Splatting training.

Your `compute_regularizer` is added directly to the photometric loss
(0.8 * L1 + 0.2 * SSIM) every training step. The goal is to design a
regularization term, computed only from the Gaussian parameters (no
ground-truth depth or normals), that improves novel-view reconstruction
quality on MipNeRF360 scenes — higher PSNR / SSIM, lower LPIPS.

Call signature:

    compute_regularizer(splats, step, scene_scale) -> scalar tensor

Where:
    splats        torch.nn.ParameterDict, keys below (first dim = N Gaussians)
    step          int, current iteration (0 .. max_steps-1)
    scene_scale   float, approx scene radius for distance normalization

splats keys (tensor shapes):
    "means"       [N, 3]      world-space Gaussian centers
    "scales"      [N, 3]      log-scales; torch.exp(...) for actual scale
    "quats"       [N, 4]      rotation quaternion (unnormalized)
    "opacities"   [N]         logit-opacities; torch.sigmoid(...) for [0,1]
    "sh0"         [N, 1, 3]   DC spherical-harmonic coefficients
    "shN"         [N, K, 3]   higher-order SH, K=(max_sh_deg+1)**2 - 1

Evaluation: PSNR (higher is better), SSIM (higher is better),
            LPIPS (lower is better). Measured on held-out views.
"""

import torch
import torch.nn.functional as F


# ============================================================================
# EDITABLE REGION — implement your regularizer in the block below.
# ============================================================================

def compute_regularizer(splats, step, scene_scale):
    """Return a scalar tensor added to the photometric loss.

    TODO: design a regularizer that improves reconstruction quality.
    Hints:
      - Parameter-level penalties (scale, opacity, SH) are cheap and often
        effective. Tune the weights per scene scale.
      - Neighbor-based priors (e.g. kNN over splats["means"]) add a small
        amount of spatial structure.
      - Keep the cost bounded: this function is called every training step.
    """
    # Default: no regularization.
    return torch.zeros((), device=splats["means"].device)

# ============================================================================
# End editable region. The training loop imports compute_regularizer from
# this file and adds its return value to the photometric loss unchanged.
# ============================================================================
