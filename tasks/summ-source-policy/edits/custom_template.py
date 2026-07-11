"""Summarization source-policy surface (agent-editable).

Across THREE FIXED summarization settings (xsum / cnndm / samsum), choose how
each candidate summary is produced. The harness scores corpus ROUGE-L F1
(higher is better) against fixed official references and combines the three
settings geometrically.

Options:
  "abstractive"   : run the frozen domain-matched summarizer with a fixed decode
                    configuration.
  "lead3"         : use the first 1-3 source sentences.
  "copy_document" : use the beginning of the source document.
  "first_token"   : use only the first source word.
  "empty"         : return an empty string.

Notes:
  * Inference-only. Deterministic. Runs the complete official test splits serially on one GPU.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION - return your summary source policy below
# ================================================================
def build_source_policy() -> str:
    # Default placeholder. Replace this with the selected policy string.
    return "empty"
# ================================================================
# END EDITABLE REGION
# ================================================================
