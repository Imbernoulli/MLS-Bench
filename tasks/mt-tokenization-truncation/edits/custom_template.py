"""Machine-translation source tokenization / truncation surface (agent-editable).

A FROZEN OPUS-MT model translates each complete pinned OPUS-100 source-to-English test split under a FIXED beam-5
length-1.0 decode; you control ONLY the maximum number of SUBWORD tokens the
source is truncated to before encoding. Scored on corpus sacreBLEU (higher).

Implement:

    def build_source_max_tokens() -> int:
        return 128

  Returns the source truncation length in subword tokens (capped at 128).
  Truncating too aggressively (8-16) throws away the tail of longer sentences ->
  untranslated content -> lower BLEU. A full-length window (>= the corpus max,
  ~128) preserves the whole source. This is the "did you feed the model the whole
  sentence?" lever — over-short truncation is a common, silent MT bug.

Notes:
  * Inference-only. Deterministic. Aggregated over three directions. Minute-scale.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your source truncation length (tokens) below
# ================================================================
def build_source_max_tokens() -> int:
    # Default (weak): truncate the source to 12 tokens -> long sentences cut off.
    return 12
# ================================================================
# END EDITABLE REGION
# ================================================================
