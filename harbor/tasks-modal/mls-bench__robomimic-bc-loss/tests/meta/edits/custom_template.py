"""
Custom BC Loss Function for GMM-based Behavioral Cloning.

This module defines the loss function used by BC-GMM training in robomimic.
The loss receives the GMM distribution produced by the policy network and
the expert demonstration actions, and returns a scalar loss.

The custom loss is imported and used by the patched BC_GMM._compute_losses
method during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


# ── Custom BC Loss Function ────────────────────────────────────────────────
# EDITABLE REGION START
class CustomBCLoss(nn.Module):
    """Custom loss for GMM-based behavioral cloning.

    Called during BC-GMM training. Receives the full GMM distribution and
    expert actions, returns a scalar loss to minimize.

    Args:
        dist: MixtureSameFamily GMM distribution (5 modes, 7-dim actions)
            Supports: .log_prob(), .sample(), .component_distribution,
                      .mixture_distribution
        target_actions: [B, 7] expert actions

    Returns:
        scalar loss tensor
    """

    def __init__(self, action_dim=7):
        super().__init__()
        self.action_dim = action_dim

    def forward(self, dist, target_actions):
        return -dist.log_prob(target_actions).mean()
# EDITABLE REGION END
