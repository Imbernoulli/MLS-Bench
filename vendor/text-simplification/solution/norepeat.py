"""Text-simplification NO-REPEAT-NGRAM surface (agent-editable, ISOLATED from beam
width — greedy decode is FIXED so this lever is visible on its own).

A FROZEN pretrained t5-base simplifier rewrites a FIXED small ASSET/TURK/WikiAuto
test slice GREEDILY (num_beams=1, repetition_penalty=1.0 both FIXED); you control
ONLY the NO-REPEAT NGRAM BLOCK SIZE. The rewrites are scored on corpus SARI (higher
is better) against the FIXED multi-reference set.

Implement:

    def build_no_repeat_ngram_size() -> int:
        return 3

`no_repeat_ngram_size` (0 = off) hard-blocks any n-gram already generated in the
sequence from recurring. A greedy T5 simplifier can otherwise loop on a repeated
n-gram indefinitely (wasting the length budget without adding new ADD/KEEP-credited
content, hurting SARI); a moderate block (n=3) removes loops without over-
constraining legitimate short repeated phrases. The DEFAULT here is the WEAK off
setting (0).

Background:
  Isolated from beam search / repetition_penalty (both FIXED off) so the effect of
  ONLY the n-gram block is visible: this is the classic greedy-decode degenerate-
  repetition failure mode and its standard hard-constraint fix.

Notes:
  * Inference-only. Deterministic. Runs on a single GPU in well under a minute.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your no-repeat n-gram size below
# ================================================================
def build_no_repeat_ngram_size() -> int:
    # Default (weak): off -> greedy decode free to loop on repeated n-grams.
    return 0
# ================================================================
# END EDITABLE REGION
# ================================================================
