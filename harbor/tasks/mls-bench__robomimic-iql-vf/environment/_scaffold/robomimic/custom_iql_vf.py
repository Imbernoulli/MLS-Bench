"""
Custom Value Function Loss for Implicit Q-Learning (IQL).

This module defines the value function loss used by IQL training in
robomimic. The loss receives predicted state values V(s), target
Q-values Q(s,a), and an asymmetry parameter (quantile/tau), and
returns a scalar loss that trains V(s) to approximate a high quantile
of the Q-value distribution.

The custom loss is imported and used by the patched IQL._compute_critic_loss
method during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Custom Value Function Loss ─────────────────────────────────────────────
# EDITABLE REGION START
def custom_vf_loss(vf_pred, q_target, quantile=0.9):
    """Custom value function loss for IQL.

    Computes an asymmetric regression loss that pushes V(s) toward a
    high quantile of Q(s,a) without explicit maximization over actions.
    The default implementation uses expectile regression (IQL paper).

    Args:
        vf_pred: [B, 1] predicted state values V(s)
        q_target: [B, 1] target Q-values Q(s,a) (detached)
        quantile: float in (0, 1), asymmetry parameter (tau)

    Returns:
        scalar loss tensor
    """
    diff = vf_pred - q_target
    weight = torch.where(diff > 0, 1.0 - quantile, quantile)
    return (weight * (diff ** 2)).mean()
# EDITABLE REGION END
