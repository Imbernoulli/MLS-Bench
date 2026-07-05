"""Text-simplification EARLY-STOPPING surface (agent-editable).

A FROZEN pretrained t5-base simplifier rewrites a FIXED small ASSET/TURK/WikiAuto
test slice with beam search (num_beams=5, no_repeat_ngram_size=3,
length_penalty=1.0, max_length=128 all FIXED); you control ONLY the beam-search
EARLY STOPPING policy. The rewrites are scored on corpus SARI (higher is better)
against the FIXED multi-reference set.

Implement:

    def build_early_stopping():
        return True

Must return one of `False`, `True`, or the string `"never"` (the three values
`transformers.generate(early_stopping=...)` accepts):
  False   : never use the "enough finished hypotheses" heuristic — keep expanding
            every beam all the way to max_length regardless of EOS. Wastes search
            budget continuing already-finished, EOS-terminated hypotheses' worse
            siblings and can let a lower-quality continuation creep into the
            returned top-1 sequence.
  True    : stop as soon as `num_beams` EOS-terminated hypotheses exist — the
            standard efficient heuristic.
  "never" : stop only when it is provably impossible to find a better hypothesis
            (the canonical, exhaustive beam-search stopping criterion).

Background:
  For a short-sequence task like sentence simplification, `True` (the standard
  heuristic) and `"never"` (the exhaustive criterion) both terminate beam search at
  essentially the right point; `False` (heuristic OFF) is the WEAK setting that
  needlessly continues searching past the point where the best hypothesis is
  already found, which can perturb which sequence rescoring finally selects as
  top-1.

Notes:
  * Inference-only. Deterministic. Runs on a single GPU in well under a minute.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your early-stopping policy below
# ================================================================
def build_early_stopping():
    # Default (weak): heuristic OFF -> needless continued search.
    return False
# ================================================================
# END EDITABLE REGION
# ================================================================
