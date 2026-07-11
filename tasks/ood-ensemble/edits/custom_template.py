"""Score-ensembling OOD score surface (agent-editable) for ood-ensemble.

Design a post-hoc OOD SCORE that ENSEMBLES multiple score FAMILIES. A single method (e.g.
deep k-NN on features alone) captures one notion of ID-ness; combining a LOGIT-space
score (free-energy) with a FEATURE-space score (deep k-NN), each STANDARDIZED on the ID
fit set and summed, captures complementary signal and is stronger than either alone
across far/near/medium OOD shifts.

    class Scorer:
        def fit(self, ctx): ...           # ctx.tr_logits, ctx.tr_feats available
        def score(self, logits, feats, early=None, raw=None):
            ...

The default below uses ONLY the k-NN feature score (a single method) -- deliberately
WEAK relative to a standardized-sum ensemble of energy + k-NN. Edit `fit`/`score` to also
compute the free-energy score on `logits`, standardize both components using statistics
fit on `ctx.tr_logits`/`ctx.tr_feats`, and sum them. Return per-sample scores as a numpy
array or torch tensor (HIGHER = more ID).
"""
from __future__ import annotations

import torch

EPS = 1e-6
K = 50


# ================================================================
# EDITABLE REGION — design the score ensemble below
# ================================================================
class Scorer:
    def fit(self, ctx):
        # Default: single-method k-NN only (WEAK -- no cross-method signal).
        self.bank = ctx.tr_feats.double()
        return self

    def _knn(self, feats):
        z = feats.double()
        out = torch.empty(z.shape[0], dtype=torch.float64)
        B = 512
        for i in range(0, z.shape[0], B):
            d = torch.cdist(z[i:i + B], self.bank)
            out[i:i + B] = -d.kthvalue(min(K, self.bank.shape[0]), dim=1).values
        return out

    def score(self, logits, feats, early=None, raw=None):
        return self._knn(feats).numpy()
# ================================================================
# END EDITABLE REGION
# ================================================================
