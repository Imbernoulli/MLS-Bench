"""Text-simplification BEAM-WIDTH (isolated) surface (agent-editable).

A FROZEN pretrained t5-base simplifier rewrites the complete pinned official
ASSET, TurkCorpus, and WikiAuto test partitions under a FIXED repetition/length
config (no_repeat_ngram_size=3,
length_penalty=1.0, max_length=128, early_stopping=True all FIXED); you control
ONLY the beam WIDTH (`num_beams`). This isolates the beam-width lever from
no_repeat_ngram_size (which simp-decoding-beam varies jointly with num_beams).

Implement:

    def build_num_beams() -> int:
        return ...

The value must be a positive integer inside the documented runtime bound.
Different widths change search behavior and compute under one fixed cap.
Invalid values fail instead of being clamped.
The benchmark measures the multi-domain result without publishing an ordering.
The native value remains runnable for no-edit verification.

Background:
  Beam search keeps a bounded set of partial hypotheses at each step.
  The submitted verifier measures how a selected width affects SARI.
  No preferred value or anchor is disclosed here.

Notes:
  * Inference-only and deterministic. Verification must generate a complete
    prediction for every official test example; no reduced path is valid.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your beam width below
# ================================================================
def build_num_beams() -> int:
    # Native no-edit value; replace it to test another width.
    return 2
# ================================================================
# END EDITABLE REGION
# ================================================================
