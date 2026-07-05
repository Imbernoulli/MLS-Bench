"""Text-simplification sampling TEMPERATURE surface (agent-editable).

A FROZEN pretrained t5-base simplifier rewrites a FIXED small ASSET/TURK/WikiAuto
test slice using SAMPLING (do_sample=True, num_beams=1, no_repeat_ngram_size=3 all
FIXED); you control ONLY the softmax TEMPERATURE applied before sampling. The
rewrites are scored on corpus SARI (higher is better) against the FIXED
multi-reference set.

Implement:

    def build_temperature() -> float:
        return 1.0

`temperature` is hard-capped to [0.05, 2.5] by the shared sanitizer.
  * HIGH temperature (>1) flattens the softmax towards uniform: sampling draws more
    unlikely / off-distribution tokens, which rarely land on a correct simplified
    phrasing -> LOWER SARI.
  * LOW temperature (<1) sharpens the softmax towards the model's mode (closer to
    greedy): sampling draws the model's preferred tokens more often -> HIGHER SARI.

Background:
  Sampling with no search (num_beams=1) always trails a proper beam decode on a
  precision-sensitive reference metric like SARI, but among sampling
  configurations, a colder (lower) temperature that stays close to the model's own
  mode is markedly better than a hot, near-random one. The DEFAULT here is a WEAK
  hot temperature (2.0) that over-randomizes the output.

Notes:
  * Inference-only. Runs on a single GPU in well under a minute. Sampling with a
    FIXED seed (`common.setup`) is deterministic run-to-run.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your sampling temperature below
# ================================================================
def build_temperature() -> float:
    # Default (weak): hot temperature -> near-random sampling.
    return 2.0
# ================================================================
# END EDITABLE REGION
# ================================================================
