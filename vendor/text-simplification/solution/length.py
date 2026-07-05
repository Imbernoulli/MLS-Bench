"""Text-simplification length / compression decode surface (agent-editable).

A FROZEN pretrained t5-base simplifier rewrites a FIXED small ASSET test slice
with beam search (num_beams=5, no-repeat-3gram FIXED); you control ONLY the
LENGTH / COMPRESSION window of the decode. The rewrites are scored on corpus SARI
(higher is better) against the FIXED multi-reference set.

Implement:

    def build_length_config() -> dict:
        return {"min_length": 0, "max_length": 96, "length_penalty": 1.0}

The three knobs (transformers `model.generate`):
  min_length     : minimum number of generated tokens.
  max_length     : maximum number of generated tokens (hard-capped at 160).
  length_penalty : beam-search length penalty. >1.0 favours LONGER sequences,
                   <1.0 favours SHORTER (more compression).

Background:
  Simplification usually SHORTENS a sentence (drops subordinate clauses,
  splits/omits complex material). SARI rewards correct DELETE edits, so length is
  a direct lever on the DELETE/ADD balance: a runaway-long decode (large
  length_penalty, large max_length) keeps everything, behaving like copy-the-input
  (few DELETE credits -> low SARI); an over-short decode drops too much (recall
  collapses). A sensibly compressive window near the reference length maximizes
  SARI. The DEFAULT here is the WEAK runaway-long config (length_penalty=2.5) that
  under-compresses.

Notes:
  * Inference-only. Deterministic. Runs on a single GPU in well under a minute.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your length / compression decode config below
# ================================================================
def build_length_config() -> dict:
    # Default (weak): runaway-long window -> under-compressed, near copy-input.
    return {"min_length": 40, "max_length": 160, "length_penalty": 2.5}
# ================================================================
# END EDITABLE REGION
# ================================================================
