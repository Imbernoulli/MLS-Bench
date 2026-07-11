"""Temperature-scaling OOD score surface (agent-editable) for ood-temperature.

Design a post-hoc OOD SCORE from the FROZEN classifier's LOGITS using the free-ENERGY
score (Liu et al., NeurIPS 2020), T*logsumexp(logits/T). The energy score's SHAPE depends
on the temperature T: at low T (T~1) the logsumexp stays close to the max logit and keeps
most of the ID/OOD logit-magnitude gap; at high T (T>>1) the softmax the energy is derived
from becomes near-uniform and the score FLATTENS, throwing away exactly the magnitude
signal that made energy beat MSP in the first place.

    class Scorer:
        def fit(self, ctx): ...
        def score(self, logits, feats, early=None, raw=None):    # HIGHER = more ID
            ...

The default below uses a very HIGH temperature (T=1000) -- deliberately WEAK: it nearly
recovers a temperature-agnostic (MSP-like) shape and loses most of the magnitude
separation. A well-chosen LOW temperature (T close to 1, matching Liu et al.'s original
recipe) is markedly stronger on every OOD shift level (far/near/medium). Return per-sample
scores as a numpy array or torch tensor (HIGHER = more ID).
"""
from __future__ import annotations

import torch


# ================================================================
# EDITABLE REGION — design the temperature-scaled energy score below
# ================================================================
class Scorer:
    def fit(self, ctx):
        return self

    def score(self, logits, feats, early=None, raw=None):
        # Default: energy at a very HIGH temperature (WEAK -- flattens the magnitude gap).
        T = 1000.0
        return (T * torch.logsumexp(logits / T, dim=-1)).numpy()
# ================================================================
# END EDITABLE REGION
# ================================================================
