"""mono3d-uncertainty-weighting WEAK baseline: DEGENERATE fixed weights (starve depth + dims).

Combine the multi-task losses with a DEGENERATE fixed weighting that all but ignores the depth
and dimension losses (weight 0.001) and puts everything on orientation. Because the geometry
depth Z = f*H/h2d depends on the predicted metric height H (from the dims head) and on the depth
head's residual, starving the depth+dims losses leaves H and the residual essentially untrained
-> the geometry depth is badly mis-scaled -> AP3D collapses. This is the deliberately-degenerate
fixed weighting. The learned homoscedastic weighting keeps every task supervised and wins.
Reference: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses" (CVPR 2018).
"""
import torch.nn as nn


def build_task_weighting():
    def weight(losses):
        # starve depth + dims (0.001), over-weight orient -> untrained H/residual, broken depth.
        return 0.001 * losses["depth"] + losses["orient"] + 0.001 * losses["dims"]

    return nn.Identity(), weight
