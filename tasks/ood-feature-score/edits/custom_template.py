"""Editable feature-only score surface for ``ood-feature-score``.

``fit`` receives ID-fit features and labels. ``score`` receives only features
and must return one finite scalar per sample, with higher values indicating ID.
"""

import torch


# ================================================================
# EDITABLE REGION - design the feature-only score below
# ================================================================
class Scorer:
    def fit(self, ctx):
        self.center = ctx.tr_feats.double().mean(dim=0)
        return self

    def score(self, feats):
        return (-(feats.double() - self.center).norm(dim=-1)).numpy()
# ================================================================
# END EDITABLE REGION
# ================================================================
