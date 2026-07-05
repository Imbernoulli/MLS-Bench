"""Text-simplification NUCLEUS (top-p) sampling surface (agent-editable).

A FROZEN pretrained t5-base simplifier rewrites a FIXED small ASSET/TURK/WikiAuto
test slice using SAMPLING (do_sample=True, num_beams=1, temperature=1.0,
no_repeat_ngram_size=3 all FIXED); you control ONLY the NUCLEUS (top-p) mass kept
before sampling. The rewrites are scored on corpus SARI (higher is better) against
the FIXED multi-reference set.

Implement:

    def build_top_p() -> float:
        return 0.6

`top_p` is hard-capped to [0.01, 1.0] by the shared sanitizer. Nucleus sampling
(Holtzman et al. 2019) keeps the smallest token set whose cumulative probability
>= top_p, renormalizes, and samples from that set only.
  * WIDE nucleus (top_p close to 1.0): samples from (nearly) the full vocabulary,
    including many low-probability / off-distribution tokens -> LOWER SARI.
  * TIGHT nucleus (top_p small): restricts sampling to only the model's most
    probable tokens -> closer to the model's mode -> HIGHER SARI.

Background:
  A tight nucleus is the standard fix for the well-known "sampling from the full
  distribution produces incoherent text" failure mode. The DEFAULT here is a WEAK
  wide nucleus (1.0, i.e. unrestricted full-distribution sampling).

Notes:
  * Inference-only. Runs on a single GPU in well under a minute. Sampling with a
    FIXED seed (`common.setup`) is deterministic run-to-run.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your nucleus (top-p) mass below
# ================================================================
def build_top_p() -> float:
    # Default (weak): wide nucleus -> unrestricted (near-full-distribution) sampling.
    return 1.0
# ================================================================
# END EDITABLE REGION
# ================================================================
