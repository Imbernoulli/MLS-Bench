"""Machine-translation post-processing / detok surface (agent-editable).

A FROZEN OPUS-MT model translates each complete pinned OPUS-100 source-to-English test split under a FIXED beam-5
length-1.0 decode; you choose a POST-PROCESSING policy applied to the model
output before scoring. Scored on corpus sacreBLEU (higher is better) against the
FIXED cased, punctuated English references.

Implement:

    def build_postproc() -> str:
        return "normalize"

Rules:
  "identity"    : model output unchanged                              [reference].
  "normalize"   : collapse repeated whitespace / strip edges          [strong].
  "lowercase"   : lowercase everything -> mismatches cased references  [degenerate].
  "strip_punct" : remove all punctuation -> loses ref punctuation n-grams [degenerate].

Background:
  The model already emits properly cased, punctuated, SentencePiece-detokenized
  English, so the right policy is a light normalization (near-identity). Lossy
  "normalizations" that DESTROY information the references keep (case, punctuation)
  tank BLEU. Order: lowercase ~ strip_punct < identity ~ normalize. The lesson:
  don't wreck the model's good detok with a bad post-processor.

Notes:
  * Inference-only. Deterministic. Aggregated over three directions. Minute-scale.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your post-processing policy below
# ================================================================
def build_postproc() -> str:
    # Default (degenerate): lowercase everything (mismatches cased references).
    return "lowercase"
# ================================================================
# END EDITABLE REGION
# ================================================================
