"""Text-simplification LENGTH-PENALTY (alpha, isolated) surface (agent-editable).

A FROZEN pretrained t5-base simplifier rewrites a FIXED small ASSET/TURK/WikiAuto
test slice under a FIXED beam width and length WINDOW (num_beams=5,
no_repeat_ngram_size=3, min_length=0, max_length=96 all FIXED); you control ONLY
the beam-search length-penalty EXPONENT `length_penalty` (alpha). This isolates
the length-NORMALIZATION exponent from the length WINDOW (max_length/min_length),
which simp-length-control varies jointly with length_penalty.

Implement:

    def build_length_penalty() -> float:
        return 1.0

Hard-capped to [0.1, 5.0] by the shared sanitizer. Beam search scores a hypothesis
by its log-probability divided by length**length_penalty (see Wu et al. 2016 GNMT):
  * alpha > 1 (e.g. 2.0) OVER-rewards longer sequences relative to their
    log-probability -> the beam search prefers padding-out / under-deleting ->
    acts closer to copy-the-input -> LOWER SARI.
  * alpha ~= 1 (neutral) or a mild alpha < 1 lets beam search compare hypotheses
    fairly without an artificial length bonus -> recovers the correct
    ADD/KEEP/DELETE edits -> HIGHER SARI.
The DEFAULT here is a WEAK large alpha (2.5) that over-rewards length.

Background:
  This is the SAME `length_penalty` generate() kwarg as simp-length-control, but
  there max_length/min_length ALSO change between weak/strong (conflating window
  size with the exponent). Here the window is FIXED (min_length=0, max_length=96,
  identical to simp-length-control's strong config) and ONLY alpha varies, so the
  exponent's effect on the length-normalization score is isolated.

NOT SHIPPED (honest finding, k1 H20, 2026-07-05, correct pinned image): once the
  length WINDOW is already fixed near-optimal (min_length=0, max_length=96), alpha
  in [1.0, 2.5] is FLAT-to-noise on turk/wiki (asset 44.71/45.03/45.14 rises mildly,
  but turk 43.76/43.70/43.68 and wiki 43.38/43.19/43.32 do NOT move monotonically --
  differences are within run-to-run noise, <0.5 SARI). No task scaffold was built
  for this surface; kept here as a documented, measured, deliberately-dropped RQ.

Notes:
  * Inference-only. Deterministic. Runs on a single GPU in well under a minute.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your length-penalty exponent (alpha) below
# ================================================================
def build_length_penalty() -> float:
    # Default (weak): large alpha -> over-rewards length, under-deletes.
    return 2.5
# ================================================================
# END EDITABLE REGION
# ================================================================
