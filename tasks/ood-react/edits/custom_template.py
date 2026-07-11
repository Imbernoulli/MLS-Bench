"""ReAct feature-clipping OOD score surface (agent-editable) for ood-react.

Design a post-hoc OOD SCORE that applies ReAct-style RECTIFIED ACTIVATIONS (Sun et al.,
NeurIPS 2021): clip the penultimate feature per-dimension at a (high) percentile of the
ID-train activations, then RECOMPUTE the logits from the clipped feature via the SAME
classifier weight/bias, and score with the free-energy score. The idea is that OOD inputs
often produce a few abnormally LARGE activations that inflate the energy score; clipping
them removes spurious over-confidence while barely touching genuine ID activations. If
the clip threshold is too AGGRESSIVE (too low a percentile), it clips useful ID signal
too and HURTS separation -- the threshold must be high enough to be a no-op for most ID
activity while still taming OOD's extreme values.

    class Scorer:
        def fit(self, ctx):                  # ctx.tr_feats [N,D] for the clip threshold
            ...
        def score(self, logits, feats, early=None, raw=None):
            # `feats` = raw penultimate features; recompute logits after clipping via
            # `self.W`/`self.b` (the frozen classifier's final-layer weight/bias).
            ...

`ctx.model` exposes the frozen classifier; use `common.classifier_weight_bias(ctx.model)`
to get its final-layer `(weight [K,D], bias [K])`. The default below clips AGGRESSIVELY at
the 90th percentile -- deliberately WEAK (clips too much genuine ID signal). A HIGHER
percentile threshold (closer to a no-op) is markedly stronger. Return per-sample scores
as a numpy array or torch tensor (HIGHER = more ID).
"""
from __future__ import annotations

import torch


# ================================================================
# EDITABLE REGION — design the ReAct clip threshold below
# ================================================================
class Scorer:
    def fit(self, ctx):
        import common
        self.W, self.b = common.classifier_weight_bias(ctx.model)
        # Default: AGGRESSIVE clip at the 90th percentile (WEAK -- over-clips ID signal).
        self.clip = torch.quantile(ctx.tr_feats.double(), 0.90, dim=0)
        return self

    def score(self, logits, feats, early=None, raw=None):
        z = torch.clamp(feats.double(), max=self.clip)
        clipped_logits = z @ self.W.double().t() + self.b.double()
        return torch.logsumexp(clipped_logits, dim=-1).numpy()
# ================================================================
# END EDITABLE REGION
# ================================================================
