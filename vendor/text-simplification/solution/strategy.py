"""Text-simplification DECODING STRATEGY surface (agent-editable).

A FROZEN pretrained t5-base simplifier rewrites a FIXED small ASSET/TURK/WikiAuto
test slice; you control ONLY the top-level DECODING STRATEGY. The rewrites are
scored on corpus SARI (higher is better) against the FIXED multi-reference set.

Implement:

    def build_strategy() -> str:
        return "beam"

Must be one of:
  "sample" : plain multinomial sampling (do_sample=True, num_beams=1,
             temperature=1.0). No search — the weak floor.
  "topp"   : nucleus sampling (do_sample=True, num_beams=1, top_p=0.9). Restricted
             sampling — better than plain sampling but still no search.
  "beam"   : deterministic beam search (num_beams=5, no_repeat_ngram_size=3). Real
             search over the sequence probability — the strong choice.

All three share the SAME fixed length window (max_length=128) and the SAME frozen
model; only the search STRATEGY varies.

Background:
  Beam search approximately maximizes sequence probability by keeping the top-k
  partial hypotheses; sampling-based decoding (with or without nucleus truncation)
  never performs this search, so it systematically under-performs beam search on a
  precision-sensitive reference metric like SARI. The DEFAULT here is the WEAK
  plain-sampling strategy.

Notes:
  * Inference-only. Runs on a single GPU in well under a minute.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your decoding strategy below
# ================================================================
def build_strategy() -> str:
    # Default (weak): plain sampling, no search at all.
    return "sample"
# ================================================================
# END EDITABLE REGION
# ================================================================
