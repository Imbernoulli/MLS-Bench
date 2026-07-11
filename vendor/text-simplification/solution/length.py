"""Text-simplification length / compression decode surface (agent-editable).

A FROZEN pretrained t5-base simplifier rewrites the complete pinned official
ASSET, TurkCorpus, and WikiAuto test partitions with beam search (num_beams=5,
no-repeat-3gram FIXED); you control ONLY the
LENGTH / COMPRESSION window of the decode. The rewrites are scored on corpus SARI
(higher is better) against the FIXED multi-reference set.

Implement:

    def build_length_config() -> dict:
        return {"min_length": ..., "max_length": ..., "length_penalty": ...}

The three knobs (transformers `model.generate`):
  min_length     : minimum number of generated tokens.
  max_length     : maximum number of generated tokens (hard-capped at 160).
  length_penalty : beam-search length penalty. >1.0 favours LONGER sequences,
                   <1.0 favours SHORTER (more compression).

Background:
  Length controls change the DELETE/ADD/KEEP balance measured by SARI.
  The mapping must contain exactly the three documented keys with valid numeric
  types and an ordered minimum/maximum pair.
  Every configuration uses the same frozen model and private references.
  No candidate values or ordering are prescribed.
  Invalid values fail rather than being clamped or repaired.
  The native mapping remains runnable for no-edit verification.
  Compare alternatives using submitted verifier results.

Notes:
  * Inference-only and deterministic. Verification must generate a complete
    prediction for every official test example; no reduced path is valid.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your length / compression decode config below
# ================================================================
def build_length_config() -> dict:
    # Native no-edit mapping; replace it to test another length configuration.
    return {"min_length": 40, "max_length": 160, "length_penalty": 2.5}
# ================================================================
# END EDITABLE REGION
# ================================================================
