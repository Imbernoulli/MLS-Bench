"""Feature-space OOD score surface (agent-editable) for ood-feature-score.

Design a post-hoc OOD SCORE from the FROZEN classifier's PENULTIMATE FEATURES. The score
maps each input's 128-dim feature vector to a scalar; by convention HIGHER = more
in-distribution (ID). The harness separates CIFAR-10 test (ID) from an OOD set (SVHN) by
AUROC and FPR@95TPR.

    class Scorer:
        def fit(self, ctx):                # ctx.tr_feats [N,D], ctx.tr_labels [N],
            ...                            # ctx.tr_logits [N,K], ctx.num_classes, feat_dim
        def score(self, logits, feats):    # feats [N,D] torch.Tensor; return [N] scores
            ...

Feature-space scores measure how close a test feature is to the ID feature manifold. The
default below is a WEAK score: the negative Euclidean distance to the GLOBAL feature mean
(a single isotropic Gaussian, no per-class structure, no whitening, no normalization).
Strong feature scores exploit ID structure, e.g.:
  * class-conditional Mahalanobis with a TIED covariance (Lee et al. 2018), ideally on
    L2-NORMALIZED features (the Mahalanobis++ trick -- normalising features onto the unit
    sphere makes the tied covariance a far better fit and sharply improves OOD detection);
  * deep k-NN: negative distance to the k-th nearest ID training embedding on L2-normalized
    features (Sun et al. 2022), a non-parametric score with no Gaussian assumption.
Fit any ID statistics you need in fit() from ctx (NO OOD data is available -- post-hoc).
Return per-sample scores as a numpy array or torch tensor (HIGHER = more ID).
"""
from __future__ import annotations

import torch

EPS = 1e-6


# ================================================================
# EDITABLE REGION — design the feature-space OOD score below
# ================================================================
class Scorer:
    def fit(self, ctx):
        # Default: fit a single global mean of the ID-train features (isotropic, weak).
        self.mu = ctx.tr_feats.double().mean(dim=0)   # [D]
        return self

    def score(self, logits, feats):
        # Default: negative Euclidean distance to the global feature mean (WEAK).
        z = feats.double()
        d = (z - self.mu).norm(dim=-1)
        return (-d).numpy()
# ================================================================
# END EDITABLE REGION
# ================================================================
