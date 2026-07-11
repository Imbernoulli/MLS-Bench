"""Summarization decoding-TEMPERATURE surface (agent-editable).

FROZEN domain-matched summarizers decode THREE FIXED domain settings (xsum / cnndm
/ samsum) with nucleus sampling (top_p=0.95) + no-repeat-3gram + the per-domain
length window FIXED; you control ONLY the sampling TEMPERATURE. Scored on corpus
ROUGE-L F1 (gmean over the 3 settings).

Implement:

    def build_temperature() -> float:
        return ...

  temperature : finite softmax temperature inside the documented runtime bound.
                It changes the sampling distribution while every other decode
                control stays fixed. Invalid values fail rather than being
                clamped.

Background:
  Compare temperatures using the complete official test splits; no
  measured ordering or preferred value is exposed in this file.

Notes:
  * Inference-only. Sampling is seeded (seed 42). Runs the complete official test splits serially on one GPU.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your sampling temperature below
# ================================================================
def build_temperature() -> float:
    # Native no-edit value; replace it to test another temperature.
    return 2.0
# ================================================================
# END EDITABLE REGION
# ================================================================
