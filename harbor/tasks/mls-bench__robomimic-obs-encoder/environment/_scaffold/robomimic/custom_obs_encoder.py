"""
Custom Observation Encoder for multi-modal robot state fusion.

This module defines the observation encoder used by BC-GMM training
in robomimic. The encoder receives a dictionary of observation tensors
(end-effector pose, gripper state, object state) and returns a fused
feature vector that is fed into the MLP backbone and GMM heads.

The custom encoder is imported and used by the patched BC_GMM network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Custom Observation Encoder ─────────────────────────────────────────────
# EDITABLE REGION START
class CustomObsEncoder(nn.Module):
    """Custom observation encoder for multi-modal robot state fusion.

    Fuses multiple observation modalities (end-effector position,
    orientation, gripper state, object state) into a single feature
    vector. The default implementation concatenates all observations.

    Observation groups:
        - robot0_eef_pos: [B, 3] end-effector position
        - robot0_eef_quat: [B, 4] end-effector quaternion orientation
        - robot0_gripper_qpos: [B, 2] gripper joint positions
        - object: [B, D_obj] object state (position, orientation, etc.)

    Args:
        obs_dims: dict mapping obs key names to their dimensions

    Returns:
        [B, output_dim] fused feature vector
    """

    def __init__(self, obs_dims):
        super().__init__()
        self.obs_dims = obs_dims
        self.output_dim = sum(obs_dims.values())

    def forward(self, obs_dict):
        parts = [obs_dict[k] for k in sorted(self.obs_dims.keys())]
        return torch.cat(parts, dim=-1)
# EDITABLE REGION END
