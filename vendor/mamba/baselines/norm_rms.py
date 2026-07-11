"""Token-local RMSNorm baseline used by the formal sequence protocol."""

import torch
import torch.nn as nn


def make_norm(d_model):
    class _RMSNorm(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(width))
            self.eps = 1e-5

        def forward(self, value):
            variance = value.float().pow(2).mean(dim=-1, keepdim=True)
            normalized = value * torch.rsqrt(variance + self.eps).to(value.dtype)
            return normalized * self.weight

    return _RMSNorm(d_model)
