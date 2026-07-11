"""Machine-translation post-processing / detok surface (agent-editable).

A FROZEN OPUS-MT model translates each complete pinned OPUS-100 source-to-English test split under a FIXED beam-5
length-1.0 decode; you choose a POST-PROCESSING policy applied to the model
output before scoring. Scored on corpus sacreBLEU (higher is better) against the
FIXED cased, punctuated English references.

Implement:

    def build_postproc() -> str:
        return "lowercase"

Rules:
  "normalize"   : collapse repeated whitespace and strip edges.
  "lowercase"   : normalize whitespace and lowercase the output.
  "strip_punct" : remove punctuation and normalize whitespace.

Background:
  Corpus metrics compare the final surface form to fixed references. The policy
  should therefore be selected for compatibility with the reference convention,
  rather than assumed to improve every corpus.

Notes:
  * Inference-only. Deterministic. Aggregated over three directions. Minute-scale.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your post-processing policy below
# ================================================================
def build_postproc() -> str:
    # Default policy; choose the transformation justified by the fixed metric.
    return "lowercase"
# ================================================================
# END EDITABLE REGION
# ================================================================
