"""Text-simplification BEAM-WIDTH (isolated) surface (agent-editable).

A FROZEN pretrained t5-base simplifier rewrites a FIXED small ASSET/TURK/WikiAuto
test slice under a FIXED repetition/length config (no_repeat_ngram_size=3,
length_penalty=1.0, max_length=128, early_stopping=True all FIXED); you control
ONLY the beam WIDTH (`num_beams`). This isolates the beam-width lever from
no_repeat_ngram_size (which simp-decoding-beam varies jointly with num_beams).

Implement:

    def build_num_beams() -> int:
        return 5

Hard-capped to [1, 12] by the shared sanitizer. A wider beam explores more
candidate hypotheses before picking the max-probability sequence -> better
recovers the correct ADD/KEEP/DELETE edits -> higher SARI, up to diminishing
returns. A narrow beam (2) under-searches relative to a properly wide one (8).
The DEFAULT here is a WEAK narrow beam (2).

Background:
  Beam search approximately maximizes sequence probability by keeping the top-k
  partial hypotheses at each step; too narrow a beam misses the better-scoring
  completions a wider beam finds.

Notes:
  * Inference-only. Deterministic. Runs on a single GPU in well under a minute.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your beam width below
# ================================================================
def build_num_beams() -> int:
    # Default (weak): narrow beam -> under-searches.
    return 2
# ================================================================
# END EDITABLE REGION
# ================================================================
