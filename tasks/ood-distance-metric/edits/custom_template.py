"""Distance-metric OOD score surface (agent-editable) for ood-distance-metric.

Design a post-hoc, non-parametric deep k-NN OOD SCORE (Sun et al., ICML 2022): the
negative distance from a test feature to its k-th nearest neighbour in the ID-train
feature bank. The CHOICE of distance metric matters: raw (unnormalized) Euclidean
distance is dominated by feature-NORM variation (which correlates weakly with
ID-ness for this classifier), while cosine similarity (Euclidean distance after
L2-normalizing both the bank and the query) measures DIRECTION, which is far more
informative of semantic class membership.

    class Scorer:
        def fit(self, ctx):                 # ctx.tr_feats [N,D] is the ID embedding bank
            ...
        def score(self, logits, feats, early=None, raw=None):
            ...

The default below uses UNNORMALIZED Euclidean k-NN distance -- deliberately WEAK. Edit it
to L2-NORMALIZE both the bank and the query features before computing distances (cosine
k-NN) for a much stronger score. Return per-sample scores as a numpy array or torch
tensor (HIGHER = more ID).
"""
from __future__ import annotations

import torch

EPS = 1e-6
K = 50


# ================================================================
# EDITABLE REGION — design the k-NN distance-metric score below
# ================================================================
class Scorer:
    def fit(self, ctx):
        # Default: RAW (unnormalized) feature bank (WEAK -- dominated by norm variation).
        self.bank = ctx.tr_feats.double()
        return self

    def score(self, logits, feats, early=None, raw=None):
        z = feats.double()
        out = torch.empty(z.shape[0], dtype=torch.float64)
        B = 512
        for i in range(0, z.shape[0], B):
            d = torch.cdist(z[i:i + B], self.bank)
            out[i:i + B] = -d.kthvalue(min(K, self.bank.shape[0]), dim=1).values
        return out.numpy()
# ================================================================
# END EDITABLE REGION
# ================================================================
