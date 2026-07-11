"""Machine-translation repetition-block surface (agent-editable).

A FROZEN OPUS-MT model translates each complete pinned OPUS-100 source-to-English test split with a FIXED beam
width (5) and length policy; you control ONLY the `no_repeat_ngram_size`
repetition block. Scored on corpus sacreBLEU (higher is better).

Implement:

    def build_norep_config() -> dict:
        return {"no_repeat_ngram_size": 3}

  no_repeat_ngram_size : forbid repeating any n-gram of this size in the output
                         (0 disables). 1-2 is too aggressive (forbids legitimate
                         repeated function words) and HURTS; 3 is the standard MT
                         value that blocks degenerate loops without harming fluent
                         output (Paulus et al. 2017; Klein et al. 2017 OpenNMT);
                         >=5 rarely triggers on short sentences.

Notes:
  * Inference-only. Deterministic. Aggregated over three directions (de->en,
    fr->en, ru->en). Runs on one small GPU in minutes.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your repetition-block config below
# ================================================================
def build_norep_config() -> dict:
    # Default (weak): an over-aggressive 1-gram block (forbids any repeated word).
    return {"no_repeat_ngram_size": 1}
# ================================================================
# END EDITABLE REGION
# ================================================================
