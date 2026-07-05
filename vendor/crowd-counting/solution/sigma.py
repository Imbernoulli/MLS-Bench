"""Agent-editable surface: the GT-DENSITY KERNEL bandwidth (sigma).

Define `gt_sigma(points, H, W)` -> the Gaussian std (in pixels) used to render the
GROUND-TRUTH density map from the point annotations. Return either a single scalar
sigma (same for every point) or a per-point array of sigmas (len == len(points)).
Everything else is FIXED; only the GT kernel changes.

An OVERSIZED FIXED sigma over-smooths dense scenes: big kernels overlap and smear
neighbouring objects together, so the target cannot resolve individual objects and the
counter mis-counts crowded regions. A geometry-ADAPTIVE k-NN kernel sets each point's
sigma from the distance to its nearest neighbours (sigma = beta * mean_kNN_dist): small
kernels where the crowd is dense (objects stay separable), larger where it is sparse ->
lower counting MAE. This is the MCNN / CSRNet geometry-adaptive kernel (beta ~ 0.3).

    def gt_sigma(points, H, W, beta=0.3, k=3):
        import numpy as np
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        n = len(pts)
        if n <= 1:
            return np.full((n,), 6.0, dtype=np.float32)
        d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        kk = min(k, n - 1)
        nn_d = np.sqrt(np.sort(d2, axis=1)[:, :kk]).mean(axis=1)
        return np.clip(beta * nn_d, 2.0, 12.0).astype(np.float32)

The DEFAULT below is the deliberately weak OVERSIZED FIXED sigma. A crashing / malformed
gt_sigma falls back to the fixed protocol sigma (6.0 px).
"""
from __future__ import annotations

import numpy as np


# ================================================================
# EDITABLE REGION — design the GT-density kernel bandwidth below
# ================================================================
def gt_sigma(points, H, W):
    # Default: oversized FIXED kernel (weak). Smears dense scenes -> higher MAE.
    return np.full((len(points),), 14.0, dtype=np.float32)
# ================================================================
# END EDITABLE REGION
# ================================================================
