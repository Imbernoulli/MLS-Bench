"""Feature-normalization OOD score surface (agent-editable) for ood-normalization.

Design a post-hoc, class-conditional Mahalanobis OOD SCORE (tied covariance, Lee et al.
2018) and choose whether to L2-NORMALIZE the penultimate features before fitting/scoring
(the "Mahalanobis++" trick). Un-normalized features have wildly varying norms across
classes and inputs, which badly distorts a single tied covariance estimate; projecting
onto the unit sphere first makes the tied-Gaussian assumption fit far better.

    class Scorer:
        def fit(self, ctx): ...
        def score(self, logits, feats, early=None, raw=None):
            ...

The default below fits/scores on RAW (non-normalized) features -- deliberately WEAK. Edit
it to L2-normalize both the fit-time features and the score-time features for a
substantially stronger score (Mahalanobis++). Return per-sample scores as a numpy array
or torch tensor (HIGHER = more ID).
"""
from __future__ import annotations

import numpy as np
import torch

EPS = 1e-6


# ================================================================
# EDITABLE REGION — design the normalization variant of Mahalanobis below
# ================================================================
class Scorer:
    def fit(self, ctx):
        # Default: RAW (non-normalized) features (WEAK -- poor tied-covariance fit).
        feats = ctx.tr_feats.double()
        labels = ctx.tr_labels
        num_classes = ctx.num_classes
        z = feats
        means = torch.zeros(num_classes, z.shape[1], dtype=torch.float64)
        centered = torch.empty_like(z)
        for k in range(num_classes):
            idx = (labels == k).nonzero(as_tuple=True)[0]
            mk = z[idx].mean(dim=0)
            means[k] = mk
            centered[idx] = z[idx] - mk
        cov = (centered.T @ centered) / z.shape[0]
        cov += EPS * torch.eye(z.shape[1], dtype=torch.float64)
        self.means = means
        self.prec = torch.as_tensor(np.linalg.inv(cov.numpy()), dtype=torch.float64)
        return self

    def score(self, logits, feats, early=None, raw=None):
        z = feats.double()
        diff = z.unsqueeze(1) - self.means.unsqueeze(0)
        m = torch.einsum("nkd,de,nke->nk", diff, self.prec, diff)
        return (-m.min(dim=1).values).numpy()
# ================================================================
# END EDITABLE REGION
# ================================================================
