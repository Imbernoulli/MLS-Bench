"""Machine-translation output-length budget surface (agent-editable).

A FROZEN OPUS-MT model translates each complete pinned OPUS-100 source-to-English test split under a FIXED beam-5
length-1.0 decode; you control ONLY the generation budget `max_new_tokens`.
Scored on corpus sacreBLEU (higher is better).

Implement:

    def build_max_new_tokens() -> int:
        return 10

  Return the maximum number of new tokens the decoder may emit, capped at 160.
  Small values can truncate translations; larger values allow more continuation
  and increase worst-case decoding work. Select the budget using the fixed corpus
  metric.

Notes:
  * Inference-only. Deterministic. Aggregated over three directions. Minute-scale.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your max_new_tokens budget below
# ================================================================
def build_max_new_tokens() -> int:
    # Initial budget; select the value justified by the fixed metric.
    return 10
# ================================================================
# END EDITABLE REGION
# ================================================================
