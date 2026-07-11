"""Editable logit-only score surface for ``ood-logit-score``.

``fit`` receives only ID-fit logits. ``score`` receives only logits and must
return one finite scalar per sample, with higher values indicating ID.
"""

import torch


# ================================================================
# EDITABLE REGION - design the logit-only score below
# ================================================================
class Scorer:
    def fit(self, ctx):
        return self

    def score(self, logits):
        return torch.softmax(logits, dim=-1).max(dim=-1).values.numpy()
# ================================================================
# END EDITABLE REGION
# ================================================================
