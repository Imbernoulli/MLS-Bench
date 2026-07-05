"""mono3d-uncertainty-weighting STRONG baseline: learned HOMOSCEDASTIC (Kendall) weighting.

Learn a per-task log-variance log_sigma_k and weight each loss by exp(-log_sigma_k) with the
+log_sigma_k regularizer: L = sum_k exp(-s_k) L_k + s_k. The network automatically down-weights
the noisier / harder task and up-weights the reliable one, and the regularizer stops sigma from
exploding. This adaptively balances the shared encoder's gradients across depth/orient/dims,
outperforming fixed equal weights. Reference: Kendall, Gal & Cipolla, "Multi-Task Learning Using
Uncertainty to Weigh Losses for Scene Geometry and Semantics" (CVPR 2018).
"""
import torch
import torch.nn as nn


def build_task_weighting():
    class _W(nn.Module):
        def __init__(self):
            super().__init__()
            self.log_sigma = nn.Parameter(torch.zeros(3))

    mod = _W()

    def weight(losses):
        s = mod.log_sigma
        keys = ["depth", "orient", "dims"]
        total = 0.0
        for i, k in enumerate(keys):
            total = total + torch.exp(-s[i]) * losses[k] + s[i]
        return total

    return mod, weight
