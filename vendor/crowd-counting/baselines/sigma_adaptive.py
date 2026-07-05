"""Good baseline for cv-count-sigma: geometry-ADAPTIVE k-NN Gaussian kernel.

Each GT point gets sigma = beta * (mean distance to its k nearest neighbours),
clamped to a sane range. In dense regions neighbours are close -> small kernels ->
objects stay separable; in sparse regions kernels grow -> stable targets. This is the
MCNN / CSRNet geometry-adaptive kernel (beta ~ 0.3), which resolves crowded scenes far
better than an oversized fixed kernel -> lower counting MAE.

Reference: Zhang et al. MCNN (CVPR 2016); Li et al. CSRNet (CVPR 2018), sec 3.2.1.
"""
import numpy as np


def gt_sigma(points, H, W):
    beta, k = 0.3, 3
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    n = len(pts)
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    if n == 1:
        return np.array([6.0], dtype=np.float32)
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    kk = min(k, n - 1)
    nn_d = np.sqrt(np.sort(d2, axis=1)[:, :kk]).mean(axis=1)
    sig = beta * nn_d
    return np.clip(sig, 2.0, 12.0).astype(np.float32)
