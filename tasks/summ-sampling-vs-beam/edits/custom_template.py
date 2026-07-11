"""Summarization decode-STRATEGY surface (agent-editable): sampling vs beam.

FROZEN domain-matched summarizers decode THREE FIXED domain settings (xsum / cnndm
/ samsum) with a FIXED per-domain length window and no-repeat-3gram; you control
ONLY the decode STRATEGY. Scored on corpus ROUGE-L F1 (gmean over the 3 settings).

Implement:

    def build_decode_strategy() -> dict:
        return {"strategy": ..., ...}

Two forms are supported:
  beam    : requires `strategy` and `num_beams`.
  sample  : requires `strategy`, `top_p`, `top_k`, and `temperature`.
            Extra or missing fields are rejected.

Background:
  Search and sampling expose different sequence distributions under the same
  frozen models. Their relative measured performance is not part of the solution interface.
  Choose a complete strategy mapping and evaluate it across every domain.
  Invalid mappings are never repaired or routed to another strategy.
  The native mapping remains runnable for no-edit verification.
  No baseline ordering is published here.

Notes:
  * Inference-only. Sampling is seeded (seed 42). Runs the complete official test splits serially on one GPU.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your decode strategy below
# ================================================================
def build_decode_strategy() -> dict:
    # Native no-edit mapping; replace it to test another strategy.
    return {"strategy": "sample", "top_p": 0.95, "top_k": 0, "temperature": 1.0}
# ================================================================
# END EDITABLE REGION
# ================================================================
