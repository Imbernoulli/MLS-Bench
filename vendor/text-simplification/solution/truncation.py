"""Text-simplification INPUT-TRUNCATION (encoder-side) surface (agent-editable).

A FROZEN pretrained t5-base simplifier rewrites a FIXED small ASSET/TURK/WikiAuto
test slice under a FIXED strong beam decode config (num_beams=5,
no_repeat_ngram_size=3, length_penalty=1.0, max_length=128, all FIXED); you control
ONLY the ENCODER-SIDE input truncation budget (the tokenizer's `max_length` /
`truncation=True` applied to the SOURCE before encoding — NOT the decoder-side
generation length). The rewrites are scored on corpus SARI (higher is better)
against the FIXED multi-reference set.

Implement:

    def build_max_input_tokens() -> int:
        return 160

Hard-capped to [8, 160] (the harness's `MAX_INPUT_TOKENS`) by the harness.
Text-simplification sources are short sentences (mean ~15-25 words for asset/turk;
WikiAuto sources run longer, up to 80 words = 100+ subword tokens): an
AGGRESSIVELY SHORT input budget silently drops the tail of longer sources before
the model ever sees it, and the model then has no way to recover the deleted
content's ADD/KEEP credit — SARI drops, especially on the longer `wiki` setting. A
generous budget (the model's real max, 160 tokens) lets every source be read in
full. The DEFAULT here is a WEAK aggressively-short budget (16 tokens).

Background:
  This isolates the ENCODER-side truncation lever from the (FIXED) decode config
  used by every other simp-* task — a distinct failure mode from length_penalty /
  max_length (those govern the OUTPUT; this governs how much of the INPUT the model
  ever sees).

Notes:
  * Inference-only. Deterministic. Runs on a single GPU in well under a minute.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your encoder-side input-truncation budget below
# ================================================================
def build_max_input_tokens() -> int:
    # Default (weak): aggressively short input budget -> silently drops tail content.
    return 16
# ================================================================
# END EDITABLE REGION
# ================================================================
