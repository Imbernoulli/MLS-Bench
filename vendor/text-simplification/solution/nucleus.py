"""Text-simplification NUCLEUS (top-p) sampling surface (agent-editable).

A FROZEN pretrained t5-base simplifier rewrites the complete pinned official
ASSET, TurkCorpus, and WikiAuto test partitions using SAMPLING (do_sample=True,
num_beams=1, temperature=1.0,
no_repeat_ngram_size=3 all FIXED); you control ONLY the NUCLEUS (top-p) mass kept
before sampling. The rewrites are scored on corpus SARI (higher is better) against
the FIXED multi-reference set.

Implement:

    def build_top_p() -> float:
        return ...

`top_p` must be finite and inside the documented probability interval. Nucleus
sampling keeps the smallest token set whose cumulative probability reaches that
mass, renormalizes, and samples from it.
Invalid values fail instead of being clamped.
All candidate values use the same frozen model and private references.
No candidate value or ordering is prescribed.
No fallback mass is substituted after a failure.

Background:
  Nucleus mass changes the token distribution under the fixed sampling pipeline.
  Compare valid values across every declared setting.
  The native value remains runnable for no-edit verification.

Notes:
  * Inference-only. Sampling with a FIXED seed (`common.setup`) is deterministic
    run-to-run. Verification must generate every official test prediction.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your nucleus (top-p) mass below
# ================================================================
def build_top_p() -> float:
    # Native no-edit value; replace it to test another nucleus mass.
    return 1.0
# ================================================================
# END EDITABLE REGION
# ================================================================
