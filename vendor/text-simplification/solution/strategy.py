"""Text-simplification DECODING STRATEGY surface (agent-editable).

A FROZEN pretrained t5-base simplifier rewrites the complete pinned official
ASSET, TurkCorpus, and WikiAuto test partitions; you control ONLY the top-level
DECODING STRATEGY. The rewrites are
scored on corpus SARI (higher is better) against the FIXED multi-reference set.

Implement:

    def build_strategy() -> str:
        return ...

Must be one of:
  "sample" : one fixed multinomial sampling configuration.
  "topp"   : one fixed nucleus sampling configuration.
  "beam"   : one fixed deterministic beam-search configuration.
  Every selector uses identical inputs, model weights, and references.
  Invalid selectors fail before generation.
  No alternate strategy is substituted after a failure.

All three share the SAME fixed length window (max_length=128) and the SAME frozen
model; only the search STRATEGY varies.

Background:
  Search and sampling generate different sequence distributions under one frozen
  model. SARI measures the resulting edits against evaluation references.
  No selector ordering is prescribed.
  The native selector remains runnable for no-edit verification.
  Compare alternatives using submitted verifier results.

Notes:
  * Inference-only. Verification must generate a complete prediction for every
    official test example; no reduced path is valid.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your decoding strategy below
# ================================================================
def build_strategy() -> str:
    # Native no-edit selector; replace it to test another strategy.
    return "sample"
# ================================================================
# END EDITABLE REGION
# ================================================================
