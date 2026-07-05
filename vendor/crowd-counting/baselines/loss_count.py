"""Good baseline for cv-count-loss: FOREGROUND-weighted MSE + count-consistency.

Plain per-pixel MSE on a density map is dominated by the huge ZERO background (in a
crowded scene the foreground is a small fraction of pixels), so the network under-shoots
the density mass and UNDER-counts. This loss (a) up-weights the FOREGROUND pixels (where
gt density > 0) so their gradient is not drowned out, and (b) adds an explicit
COUNT-CONSISTENCY term on the integrated mass. Both directly target what the counting
metric measures -> lower counting MAE. This is the spirit of density-weighted /
count-aware crowd-counting losses (DM-Count-style explicit count supervision, Wang et
al. NeurIPS 2020).
"""


def density_loss(pred, gt):
    import torch
    import torch.nn.functional as F
    # foreground-weighted pixel loss: weight = 1 + alpha * (gt > 0)
    fg = (gt > 1e-6).float()
    w = 1.0 + 9.0 * fg
    px = (w * (pred - gt) ** 2).mean()
    # explicit count-consistency on the integrated mass
    pc = pred.sum(dim=(-2, -1)); gc = gt.sum(dim=(-2, -1))
    count_term = (pc - gc).abs().mean()
    return px + 0.02 * count_term
