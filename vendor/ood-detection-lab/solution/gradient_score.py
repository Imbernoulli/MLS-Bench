"""Editable gradient-representation OOD score for ``ood-gradient``.

``fit`` receives the complete ID-fit gradient representations. ``score``
receives the same ten-dimensional representation for evaluation samples and
must return one finite scalar per sample, with higher values indicating ID.
"""

import torch


# ================================================================
# EDITABLE REGION - design the gradient-representation score below
# ================================================================
class Scorer:
    def fit(self, ctx):
        self.center = ctx.tr_gradients.double().mean(dim=0)
        return self

    def score(self, gradients):
        return (-(gradients.double() - self.center).norm(dim=-1)).numpy()
# ================================================================
# END EDITABLE REGION
# ================================================================
