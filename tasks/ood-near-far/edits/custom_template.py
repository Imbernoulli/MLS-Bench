"""General post-hoc OOD score surface (agent-editable) for ood-near-far.

Design a post-hoc OOD SCORE using ANY of the FROZEN classifier's outputs -- logits and/or
penultimate features. The score maps each input to a scalar; by convention HIGHER = more
in-distribution (ID). This task scores the SAME score design against a NEAR-OOD set
(CIFAR-100, semantically close to CIFAR-10) where logit-only scores struggle and
feature-space structure matters more.

    class Scorer:
        def fit(self, ctx):                # ctx.tr_logits/tr_feats/tr_labels, num_classes
            ...
        def score(self, logits, feats):    # logits [N,K], feats [N,D]; return [N] scores
            ...

The default below is MSP (Maximum Softmax Probability) -- a WEAK baseline that is
especially poor on NEAR-OOD, where CIFAR-100 images look enough like CIFAR-10 that the
softmax stays confident. Stronger designs combine logit magnitude (energy) with
feature-space distance (Mahalanobis++ / deep k-NN on L2-normalized features), or use a
relative Mahalanobis score. Fit any ID statistics you need from ctx (post-hoc, OOD-free).
Return per-sample scores as a numpy array or torch tensor (HIGHER = more ID).
"""
from __future__ import annotations

import torch

EPS = 1e-6


# ================================================================
# EDITABLE REGION — design the OOD score below (logits and/or feats)
# ================================================================
class Scorer:
    def fit(self, ctx):
        return self

    def score(self, logits, feats):
        # Default: MSP -- maximum softmax probability (WEAK, especially on near-OOD).
        probs = torch.softmax(logits, dim=-1)
        return probs.max(dim=-1).values.numpy()
# ================================================================
# END EDITABLE REGION
# ================================================================
