"""Summarization decode-STRATEGY surface (agent-editable): sampling vs beam.

FROZEN domain-matched summarizers decode THREE FIXED domain settings (xsum / cnndm
/ samsum) with a FIXED per-domain length window and no-repeat-3gram; you control
ONLY the decode STRATEGY. Uses mean per-example ROUGE-L F1 (gmean over 3 settings).

Implement:

    def build_decode_strategy() -> dict:
        return {"strategy": ..., ...}

Two forms are supported:
  beam    : exactly `strategy` and integer `num_beams` in [1, 12].
  sample  : exactly `strategy`, finite `top_p` in (0, 1], integer `top_k`
            in [0, 1000], and finite `temperature` in (0, 5].

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
