"""Custom latent normalization for TD-MPC2 world model.

Replace the body of CustomSimNorm with your normalization implementation.
The class is used as the final activation in the encoder and dynamics
networks, constraining the latent representation geometry.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# EDITABLE: Custom latent normalization
# =====================================================================
class CustomSimNorm(nn.Module):
    """Custom normalization for latent state representations in world models.

    Interface contract (same as SimNorm):
        __init__(cfg)  -- cfg.simnorm_dim is the group size (default: 8)
        forward(x: Tensor) -> Tensor  (same shape as input)

    The input tensor has shape (*batch_dims, latent_dim) where latent_dim
    is divisible by simnorm_dim. Your normalization should constrain the
    geometry of the latent space to improve world model learning.

    Evaluated on DMControl walker-walk and cheetah-run tasks.
    """

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.simnorm_dim

    def forward(self, x):
        # Default: SimNorm (simplicial normalization)
        # Reshape into groups of size self.dim and apply softmax
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"CustomSimNorm(dim={self.dim})"
