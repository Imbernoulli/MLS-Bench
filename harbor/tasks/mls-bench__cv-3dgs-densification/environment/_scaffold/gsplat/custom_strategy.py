"""Custom densification strategy for 3D Gaussian Splatting.

This file defines the CustomStrategy class that controls how Gaussians
are added, split, and pruned during per-scene optimization.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import math
import torch
from gsplat.strategy.base import Strategy
from gsplat.strategy.ops import (
    duplicate, split, remove, reset_opa,
    relocate, sample_add, inject_noise_to_position,
)


# ============================================================================
# Densification Strategy (EDITABLE REGION: lines 20-90)
# ============================================================================

@dataclass
class CustomStrategy(Strategy):
    """Custom 3DGS densification strategy.

    TODO: Design your densification strategy to maximize novel view quality.

    Available operations (from gsplat.strategy.ops):
        duplicate(params, optimizers, state, mask)
            Clone Gaussians selected by boolean mask.
        split(params, optimizers, state, mask, revised_opacity=False)
            Split Gaussians into 2 new ones sampled from covariance.
        remove(params, optimizers, state, mask)
            Remove Gaussians selected by boolean mask.
        reset_opa(params, optimizers, state, value)
            Reset all opacities to logit(value).
        relocate(params, optimizers, state, mask, binoms, min_opacity)
            Teleport dead Gaussians to high-opacity locations.
        sample_add(params, optimizers, state, n, binoms, min_opacity)
            Add n new Gaussians sampled from opacity distribution.
        inject_noise_to_position(params, optimizers, state, scaler)
            Perturb Gaussian positions with noise scaled by opacity.

    Available info from rasterization (in step_pre/post_backward):
        info["means2d"]       - 2D projected means, call .retain_grad()
        info["means2d"].grad  - gradient w.r.t. 2D means (after backward)
        info["width"], info["height"] - image dimensions
        info["n_cameras"]     - number of cameras in batch
        info["radii"]         - screen-space radii [C, N, 1] or [C, N, 2]
        info["gaussian_ids"]  - visible Gaussian indices

    Available params:
        params["means"]      - [N, 3] world-space positions
        params["scales"]     - [N, 3] log-scales (use torch.exp for actual)
        params["quats"]      - [N, 4] rotation quaternions
        params["opacities"]  - [N] logit-opacities (use torch.sigmoid)
        params["sh0"]        - [N, 1, 3] DC spherical harmonics
        params["shN"]        - [N, K, 3] higher-order SH coefficients

    Evaluation metrics (for reference):
        PSNR (higher is better), SSIM (higher is better), LPIPS (lower is better)
    """

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy."""
        raise NotImplementedError("Implement initialize_state")

    def check_sanity(self, params, optimizers):
        """Sanity check for required parameters."""
        super().check_sanity(params, optimizers)
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(self, params, optimizers, state, step, info):
        """Called BEFORE loss.backward(). Retain gradients for densification."""
        raise NotImplementedError("Implement step_pre_backward")

    def step_post_backward(self, params, optimizers, state, step, info,
                           packed=False):
        """Called AFTER loss.backward(). Implement densification logic here."""
        raise NotImplementedError("Implement step_post_backward")
