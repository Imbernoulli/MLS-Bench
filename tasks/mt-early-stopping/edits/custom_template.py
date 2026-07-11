"""Machine-translation beam early-stopping surface (agent-editable).

A FROZEN OPUS-MT model translates each complete pinned OPUS-100 source-to-English test split under beam-5 with a
fixed short-biased length policy (length_penalty=0.6); you control ONLY the
beam-search STOPPING policy. Scored on corpus
sacreBLEU (higher is better).

Implement:

    def build_early_stopping():
        return "never"

Options (HF `generate` early_stopping):
  "never" : canonical stopping — only stop when NO better hypothesis can exist
            given the length penalty (Huang et al. 2017 "When to Finish?").
  False   : heuristic stop when a better hypothesis is unlikely.
  True    : stop as soon as `num_beams` finished hypotheses exist.

The stopping rule interacts with the fixed beam width and length penalty. Select
the criterion justified by the resulting corpus metric.

Notes:
  * Inference-only. Deterministic. Aggregated over three directions. Minute-scale.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your early_stopping policy below
# ================================================================
def build_early_stopping():
    # Initial policy; select the criterion justified by the fixed metric.
    return "never"
# ================================================================
# END EDITABLE REGION
# ================================================================
