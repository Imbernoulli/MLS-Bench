"""Weak baseline for cv-count-loss: plain pixel-wise MSE only.

L = mean( (pred - gt)^2 ) over density pixels. Pure L2 on the (blurry, Gaussian-
smoothed) density target, no count-level supervision. This is the MCNN/CSRNet default
loss but WITHOUT the count-consistency term; the total integrated mass is only weakly
constrained, so per-image counts drift -> higher counting MAE.
"""


def density_loss(pred, gt):
    import torch.nn.functional as F
    return F.mse_loss(pred, gt)
