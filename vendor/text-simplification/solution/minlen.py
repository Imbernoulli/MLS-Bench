"""Text-simplification MIN-LENGTH FLOOR (isolated) surface (agent-editable).

A FROZEN pretrained t5-base simplifier rewrites a FIXED small ASSET/TURK/WikiAuto
test slice under a FIXED beam width, length-penalty and max_length (num_beams=5,
no_repeat_ngram_size=3, length_penalty=1.0, max_length=96 all FIXED); you control
ONLY the decoder-side `min_length` FLOOR (the minimum number of generated
tokens). This isolates the min-length floor from the length WINDOW's upper bound
(max_length) and from length_penalty (varied in isolation by simp-length-penalty).

Implement:

    def build_min_length() -> int:
        return 0

Hard-capped to [0, 96] (clamped to max_length) by the shared sanitizer. A large
min_length FLOOR forces the decoder to keep generating past the point where it
would naturally stop (natural EOS), padding the output with low-value continuation
tokens that rarely correspond to a correct ADD/KEEP edit -> pushes the sequence
away from a genuine compressive simplification -> LOWER SARI. A small/zero floor
lets the model stop wherever beam search naturally finds the best EOS -> HIGHER
SARI (matches simp-length-control's tuned config, min_length=0).

Background:
  This is the SAME `min_length` generate() kwarg exposed jointly with max_length /
  length_penalty by simp-length-control; here max_length and length_penalty are
  FIXED at their SARI-optimal values (96 / 1.0) and ONLY the floor varies, so the
  floor's effect is isolated from the window's ceiling.

Notes:
  * Inference-only. Deterministic. Runs on a single GPU in well under a minute.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your min-length floor below
# ================================================================
def build_min_length() -> int:
    # Default (weak): large floor -> forces padding past the natural EOS.
    return 60
# ================================================================
# END EDITABLE REGION
# ================================================================
