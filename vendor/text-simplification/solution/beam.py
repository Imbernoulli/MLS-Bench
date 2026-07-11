"""Text-simplification beam / repetition decode surface (agent-editable).

A FROZEN pretrained t5-base simplifier rewrites the complete pinned official
ASSET, TurkCorpus, and WikiAuto test partitions;
you control ONLY the BEAM / REPETITION config of the decode (the length window is
FIXED). The rewrites are scored on corpus SARI (higher is better) against the
FIXED multi-reference set.

Implement:

    def build_beam_config() -> dict:
        return {"num_beams": ..., "no_repeat_ngram_size": ..., "repetition_penalty": ...}

The three knobs (transformers `model.generate`):
  num_beams            : bounded positive integer controlling search width.
                         No candidate ordering is prescribed.
                         Invalid values fail instead of being clamped.
  no_repeat_ngram_size : bounded non-negative repeated-span control.
                         Zero disables the constraint.
  repetition_penalty   : >1.0 discourages token repetition; too high distorts the
                         rewrite.

Background:
  Search and repetition controls change the generated sequence distribution.
  Every valid mapping is evaluated on the same official test partitions.
  Measured ordering and preferred values are intentionally omitted here.
  The native mapping remains runnable for no-edit verification.

Notes:
  * Inference-only and deterministic. Verification must generate a complete
    prediction for every official test example; no reduced path is valid.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your beam / repetition decode config below
# ================================================================
def build_beam_config() -> dict:
    # Native no-edit mapping; replace it to test another beam configuration.
    return {"num_beams": 1, "no_repeat_ngram_size": 0, "repetition_penalty": 1.0}
# ================================================================
# END EDITABLE REGION
# ================================================================
