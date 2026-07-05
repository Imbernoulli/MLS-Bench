"""Text-simplification REPETITION-PENALTY surface (agent-editable, ISOLATED from
beam width — greedy decode is FIXED so this lever is visible on its own).

A FROZEN pretrained t5-base simplifier rewrites a FIXED small ASSET/TURK/WikiAuto
test slice GREEDILY (num_beams=1, no_repeat_ngram_size=0 both FIXED); you control
ONLY the REPETITION PENALTY applied to previously generated tokens' logits. The
rewrites are scored on corpus SARI (higher is better) against the FIXED
multi-reference set.

Implement:

    def build_repetition_penalty() -> float:
        return 1.3

`repetition_penalty` is hard-capped to a sane range by the shared sanitizer
(effectively [1.0, ~10] in practice; values <1.0 would REWARD repetition and are
clamped away from that regime upstream). 1.0 = off (Keskar et al. 2019 CTRL-style
penalty). A greedy T5 simplifier can loop on a frequent function word without this
penalty (wasting the length budget without adding new ADD/KEEP-credited content,
hurting SARI); a moderate penalty (~1.2-1.5) breaks loops. The DEFAULT here is the
WEAK off setting (1.0).

Background:
  Isolated from beam search / n-gram blocking (both FIXED off) so the effect of
  ONLY the repetition penalty is visible: this is the classic greedy-decode
  degenerate-repetition failure mode and its standard logit-level fix.

Notes:
  * Inference-only. Deterministic. Runs on a single GPU in well under a minute.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your repetition penalty below
# ================================================================
def build_repetition_penalty() -> float:
    # Default (weak): off -> greedy decode free to loop on repeated tokens.
    return 1.0
# ================================================================
# END EDITABLE REGION
# ================================================================
