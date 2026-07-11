"""ODIN input-preprocessing OOD score surface (agent-editable) for ood-input-preproc.

Design a post-hoc OOD SCORE that uses ODIN's INPUT PREPROCESSING (Liang et al., ICLR
2018): perturb the raw input by a small step along the sign of the gradient of the
predicted-class loss, then re-run the forward pass and score the PERTURBED logits. ID
inputs sit in flatter loss regions near the decision boundary of their true class, so this
perturbation increases their confidence MORE than it does for OOD inputs -- widening the
ID/OOD gap beyond what a fixed forward pass gives.

    class Scorer:
        def fit(self, ctx): ...          # ctx.model, ctx.device also available
        def score(self, logits, feats, early=None, raw=None):
            # `raw` is the raw (CIFAR-normalized) input batch [N,3,32,32]; use
            # ctx.model directly (stored on self during fit) to recompute perturbed
            # logits via gradient ascent on the input.
            ...

The default below does NO input perturbation (epsilon=0, plain forward pass, i.e. the
free-energy score with no ODIN preprocessing) -- deliberately WEAK relative to a properly
tuned small perturbation step. Use `common.odin_preprocess_logits(model, raw, temperature,
epsilon)` (importable from the harness's `common` module, already on `sys.path`) to get
temperature-scaled PERTURBED logits, then score them (e.g. with the energy score). Return
per-sample scores as a numpy array or torch tensor (HIGHER = more ID).
"""
from __future__ import annotations

import torch


# ================================================================
# EDITABLE REGION — design the ODIN input-preprocessing score below
# ================================================================
class Scorer:
    def fit(self, ctx):
        self.model = ctx.model
        return self

    def score(self, logits, feats, early=None, raw=None):
        # Default: NO input perturbation (epsilon=0) -- WEAK, a plain energy score.
        import common
        perturbed_logits = common.odin_preprocess_logits(
            self.model, raw, temperature=1.0, epsilon=0.0)
        return torch.logsumexp(perturbed_logits, dim=-1).numpy()
# ================================================================
# END EDITABLE REGION
# ================================================================
