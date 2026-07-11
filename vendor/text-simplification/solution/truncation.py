"""Text-simplification INPUT-TRUNCATION (encoder-side) surface (agent-editable).

A FROZEN pretrained t5-base simplifier rewrites the complete pinned official
ASSET, TurkCorpus, and WikiAuto test partitions under one fixed decode
configuration shared by every candidate;
you control
ONLY the ENCODER-SIDE input truncation budget (the tokenizer's `max_length` /
`truncation=True` applied to the SOURCE before encoding — NOT the decoder-side
generation length). The rewrites are scored on corpus SARI (higher is better)
against the FIXED multi-reference set.

Implement:

    def build_max_input_tokens() -> int:
        return ...

The value must be an integer inside the documented encoder bound; invalid values
fail instead of being clamped. Input truncation changes how much source content
reaches the frozen model while the decoder remains fixed.
All candidate budgets are evaluated on the same three official source partitions
and verifier-mounted references. No candidate ordering is prescribed.
No fallback budget is substituted after a failure.
The native value remains runnable for no-edit verification.
Use submitted verifier results to compare alternatives.

Background:
  This isolates the ENCODER-side truncation lever from the (FIXED) decode config
  used by every other simp-* task — a distinct failure mode from length_penalty /
  max_length (those govern the OUTPUT; this governs how much of the INPUT the model
  ever sees).

Notes:
  * Inference-only and deterministic. Verification must generate a complete
    prediction for every official test example; no reduced path is valid.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your encoder-side input-truncation budget below
# ================================================================
def build_max_input_tokens() -> int:
    # Native no-edit value; replace it to test another input budget.
    return 16
# ================================================================
# END EDITABLE REGION
# ================================================================
