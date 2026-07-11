"""Text-simplification MIN-LENGTH FLOOR (isolated) surface (agent-editable).

A FROZEN pretrained t5-base simplifier rewrites the complete pinned official
ASSET, TurkCorpus, and WikiAuto test partitions under one fixed beam,
repetition, penalty, and maximum-length setup;
you control
ONLY the decoder-side `min_length` FLOOR (the minimum number of generated
tokens). This isolates the min-length floor from the length WINDOW's upper bound
(max_length) and from length_penalty (varied in isolation by simp-length-penalty).

Implement:

    def build_min_length() -> int:
        return ...

The value must be a non-negative integer no larger than the fixed maximum.
Invalid values fail instead of being clamped. The floor changes which stopping
points are available while every other decode control remains fixed.
All candidate values are evaluated against the same private references.
No candidate value or ordering is prescribed.
No fallback floor is substituted after a failure.
The native value remains runnable for no-edit verification.

Background:
  This is the SAME `min_length` generate() kwarg exposed jointly with max_length /
  length_penalty by simp-length-control; here max_length and length_penalty are
  fixed identically for every candidate and only the floor varies, so the
  floor's effect is isolated from the window's ceiling.

Notes:
  * Inference-only and deterministic. Verification must generate a complete
    prediction for every official test example; no reduced path is valid.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your min-length floor below
# ================================================================
def build_min_length() -> int:
    # Native no-edit value; replace it to test another floor.
    return 60
# ================================================================
# END EDITABLE REGION
# ================================================================
