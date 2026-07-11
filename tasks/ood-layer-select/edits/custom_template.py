"""Feature-layer-selection OOD score surface (agent-editable) for ood-layer-select.

Design a post-hoc, feature-space OOD SCORE (Mahalanobis++: class-conditional tied
covariance on L2-normalized features, Lee et al. 2018) and choose WHICH layer's features
to use. The frozen classifier exposes an EARLY (64-dim, low-level texture/color) feature
via `early` and the usual 128-dim PENULTIMATE (semantic) feature via `feats`.

    class Scorer:
        def fit(self, ctx):                 # ctx.tr_early [N,64] also available
            ...
        def score(self, logits, feats, early=None, raw=None):
            ...

Using ONLY the penultimate feature is a reasonable default, but it is not universally
best: concatenating the early feature with the penultimate feature gives Mahalanobis++
more low-level statistics to draw on and is stronger across far/near/medium OOD shifts.
The default below is the WEAK penultimate-ONLY design; edit it to fit/score on the
CONCATENATION of `early` and `feats` for a stronger score. Return per-sample scores as a
numpy array or torch tensor (HIGHER = more ID).
"""
from __future__ import annotations

import numpy as np
import torch

EPS = 1e-6


def _l2norm(x, eps=EPS):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


# ================================================================
# EDITABLE REGION — design the layer-selection Mahalanobis++ score below
# ================================================================
class Scorer:
    def fit(self, ctx):
        # Default: penultimate-ONLY features (WEAK -- loses low-level structure).
        feats = ctx.tr_feats.double()
        labels = ctx.tr_labels
        num_classes = ctx.num_classes
        z = _l2norm(feats)
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
        z = _l2norm(feats.double())
        diff = z.unsqueeze(1) - self.means.unsqueeze(0)
        m = torch.einsum("nkd,de,nke->nk", diff, self.prec, diff)
        return (-m.min(dim=1).values).numpy()
# ================================================================
# END EDITABLE REGION
# ================================================================
