"""Summarization no-repeat-ngram surface (agent-editable).

FROZEN domain-matched summarizers decode THREE FIXED domain settings (xsum / cnndm
/ samsum) with a FIXED beam width and per-domain length window; you control ONLY
the no_repeat_ngram_size BLOCK. Uses mean per-example ROUGE-L F1 (gmean over 3
settings).

Implement:

    def build_norepeat_size() -> int:
        return ...

  no_repeat_ngram_size : integer in [0, 20] selecting the forbidden n-gram size.
                         Zero disables the block. Positive values select the
                         forbidden span size. The benchmark measures their
                         multi-domain effect without publishing an ordering
                         here.

Background:
  Choose an integer in [0, 20] and evaluate it empirically.
  The native value remains available for no-edit verification.

Notes:
  * Inference-only. Deterministic. Runs the complete official test splits serially on one GPU.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your no-repeat-ngram size below
# ================================================================
def build_norepeat_size() -> int:
    # Native no-edit value; replace it to test another block size.
    return 0
# ================================================================
# END EDITABLE REGION
# ================================================================
