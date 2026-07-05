"""Text-simplification beam / repetition decode surface (agent-editable).

A FROZEN pretrained t5-base simplifier rewrites a FIXED small ASSET test slice;
you control ONLY the BEAM / REPETITION config of the decode (the length window is
FIXED). The rewrites are scored on corpus SARI (higher is better) against the
FIXED multi-reference set.

Implement:

    def build_beam_config() -> dict:
        return {"num_beams": 5, "no_repeat_ngram_size": 3, "repetition_penalty": 1.0}

The three knobs (transformers `model.generate`):
  num_beams            : beam width. Greedy (1) under-searches and leaves SARI on
                         the table; a tuned beam (4-6) is the standard strong
                         simplification decode. Hard-capped at 12.
  no_repeat_ngram_size : block repeated n-grams (0 = off). A small value (2-3)
                         avoids degenerate repetition without hurting real edits.
  repetition_penalty   : >1.0 discourages token repetition; too high distorts the
                         rewrite.

Background:
  Beam search is the standard lever for extractive/abstractive decode quality.
  For simplification, a real beam decode adds the right simpler words and deletes
  the right complex ones (both credited by SARI). The DEFAULT here is a WEAK
  greedy config (num_beams=1, no repetition control).

Notes:
  * Inference-only. Deterministic. Runs on a single GPU in well under a minute.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your beam / repetition decode config below
# ================================================================
def build_beam_config() -> dict:
    # Default (weak): greedy, no repetition control.
    return {"num_beams": 1, "no_repeat_ngram_size": 0, "repetition_penalty": 1.0}
# ================================================================
# END EDITABLE REGION
# ================================================================
