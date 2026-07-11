"""Regularization surface (agent-editable) for nli-regularization.

A DistilBERT cross-encoder is trained on the complete labeled SNLI train split.
You control head dropout together with global AdamW weight decay applied to
both encoder and classifier-head parameters.

Implement:

    def build_reg() -> dict:
        return {"reg": ...}

Options:
  reg : "standard" -> dropout 0.1 and weight decay 0.01.
        "none"     -> dropout 0.0 and weight decay 0.0.
        "heavy"    -> dropout 0.7 and weight decay 0.3.
                      Every setting uses the same optimizer-step inventory.

All settings share the complete corpus, encoder revision, optimizer class, seed,
and evaluation domains. The verifier measures their effect without publishing a
preferred regularization level in the editable workspace.

Notes:
  * One deterministic three-epoch full-corpus run is followed by three evaluations.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your regularization configuration below
# ================================================================
def build_reg() -> dict:
    # Native configuration; replace it to study another supported level.
    return {"reg": "heavy"}
# ================================================================
# END EDITABLE REGION
# ================================================================
