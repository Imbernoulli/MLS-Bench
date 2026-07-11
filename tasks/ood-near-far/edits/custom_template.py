"""Editable joint logit/feature score surface for ``ood-near-far``.

``fit`` receives ID-fit logits, features, and labels. ``score`` must return one
finite scalar per sample, with higher values indicating ID.
"""

import torch


# ================================================================
# EDITABLE REGION - design the joint OOD score below
# ================================================================
class Scorer:
    def fit(self, ctx):
        return self

    def score(self, logits, feats):
        return torch.softmax(logits, dim=-1).max(dim=-1).values.numpy()
# ================================================================
# END EDITABLE REGION
# ================================================================
