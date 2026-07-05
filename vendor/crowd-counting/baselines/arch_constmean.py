"""Degenerate control for cv-count-architecture: CONSTANT-MEAN predictor.

Ignores the image entirely: a single learned bias produces a UNIFORM density map, so
the integrated count is the same for every image (it converges to the training-set mean
count). On the count-EXTRAPOLATION val split (val counts are higher than train) this
scores ~ CONST_MEAN_MAE -> far worse than any real image-conditioned counter. This is
the degenerate that MUST lose.
"""


def build_counter():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ConstMean(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            B, _, H, W = x.shape
            h, w = H // 8, W // 8
            val = F.softplus(self.bias)
            return val.expand(B, h, w)   # image-independent uniform density

    return ConstMean()
