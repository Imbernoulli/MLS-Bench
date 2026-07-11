"""Machine-translation output-length budget surface (agent-editable).

A FROZEN OPUS-MT model translates each complete pinned OPUS-100 source-to-English test split under a FIXED beam-5
length-1.0 decode; you control ONLY the generation budget `max_new_tokens`.
Scored on corpus sacreBLEU (higher is better).

Implement:

    def build_max_new_tokens() -> int:
        return 128

  Returns the maximum number of NEW tokens the decoder may emit (capped at 160).
  Too tight a budget (8-16) TRUNCATES the translation -> the tail of the English
  sentence is cut off -> the brevity penalty and lost n-grams tank BLEU. A budget
  >= the natural target length (~64-128) lets the model finish. This is the "did
  you give the decoder enough room to finish?" lever.

Notes:
  * Inference-only. Deterministic. Aggregated over three directions. Minute-scale.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your max_new_tokens budget below
# ================================================================
def build_max_new_tokens() -> int:
    # Default (weak): a 10-token budget -> translations get cut off.
    return 10
# ================================================================
# END EDITABLE REGION
# ================================================================
