"""Text-simplification sampling TEMPERATURE surface (agent-editable).

A FROZEN pretrained t5-base simplifier rewrites the complete pinned official
ASSET, TurkCorpus, and WikiAuto test partitions using SAMPLING (do_sample=True,
num_beams=1, no_repeat_ngram_size=3 all FIXED); you control ONLY the softmax
TEMPERATURE applied before sampling. The
rewrites are scored on corpus SARI (higher is better) against the FIXED
multi-reference set.

Implement:

    def build_temperature() -> float:
        return ...

`temperature` must be finite and inside the documented runtime bound.
It rescales sampling logits while all other generation controls stay fixed.
Invalid values fail instead of being clamped.
The benchmark evaluates its effect on complete predictions.
No candidate value or ordering is prescribed.
No alternate temperature is substituted after a failure.

Background:
  Sampling changes the output distribution under the same frozen model.
  Compare valid temperatures on every official source partition.
  No measured ordering, preferred value, or anchor is disclosed here.
  The native value remains runnable for no-edit verification.
  Runtime failures abort the setting.

Notes:
  * Inference-only. Sampling with a FIXED seed (`common.setup`) is deterministic
    run-to-run. Verification must generate every official test prediction.
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
