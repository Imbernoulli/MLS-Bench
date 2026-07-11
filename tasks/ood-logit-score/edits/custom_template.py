"""Editable logit-only score surface for ``ood-logit-score``.

``fit`` receives all 50,000 ID-train logits; ``score`` receives only logits.
Multiple fixed OOD evaluations run outside the instruction and all contribute.
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
