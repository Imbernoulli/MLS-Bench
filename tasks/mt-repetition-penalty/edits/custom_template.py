"""Machine-translation repetition-penalty surface (agent-editable).

A FROZEN OPUS-MT model translates each complete pinned OPUS-100 source-to-English test split with a FIXED beam
width (5) and length policy; you control ONLY the `repetition_penalty`. Scored on
corpus sacreBLEU (higher is better).

Implement:

    def build_reppen_config() -> dict:
        return {"repetition_penalty": 1.8}

  repetition_penalty : divides the logits of already-generated tokens by this
                       factor (Keskar et al. 2019 CTRL). 1.0 = off. This OPUS-MT
                       model tends to OVER-GENERATE / repeat under plain beam, so
                       discouraging repeats HELPS: a mild penalty (~1.1) recovers
                       a little, and a strong penalty (~1.8) recovers more on this
                       model + data. Measured order for this setting:
                       off (1.0) < tuned (1.1) < high (1.8). (Tune it — the
                       optimum is model- and data-specific; on cleaner data a
                       penalty this high would hurt.)

Notes:
  * Inference-only. Deterministic. Aggregated over three directions. Minute-scale.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your repetition-penalty config below
# ================================================================
def build_reppen_config() -> dict:
    # Default (weak): repetition penalty OFF (1.0) -> the model over-generates.
    return {"repetition_penalty": 1.0}
# ================================================================
# END EDITABLE REGION
# ================================================================
