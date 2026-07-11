"""Machine-translation beam early-stopping surface (agent-editable).

A FROZEN OPUS-MT model translates each complete pinned OPUS-100 source-to-English test split under beam-5 with a
short-biased length policy (length_penalty=0.6, the tuned MT optimum for this
model); you control ONLY the beam-search STOPPING policy. Scored on corpus
sacreBLEU (higher is better).

Implement:

    def build_early_stopping():
        return True

Options (HF `generate` early_stopping):
  "never" : canonical stopping — only stop when NO better hypothesis can exist
            given the length_penalty (Huang et al. 2017 "When to Finish?"). It is
            the length-penalty-aware criterion, but combined with this already-
            short-biased length_penalty it OVER-searches long, worse hypotheses ->
            weakest here.
  False   : heuristic stop (stop when a better hypothesis is unlikely) — middle.
  True    : stop as soon as `num_beams` finished hypotheses exist. Under a short-
            biased length_penalty this crisp early stop lands on the tuned length
            and scores HIGHEST for this model/data. Measured order: never < False
            < True (a genuine MINOR lever; the gaps are a few BLEU).

Notes:
  * Inference-only. Deterministic. Aggregated over three directions. Minute-scale.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your early_stopping policy below
# ================================================================
def build_early_stopping():
    # Default (weak): canonical 'never' stopping — worst under this short-biased
    # length policy (stops too late relative to the tuned optimum).
    return "never"
# ================================================================
# END EDITABLE REGION
# ================================================================
