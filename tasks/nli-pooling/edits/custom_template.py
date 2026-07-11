"""Pooling surface (agent-editable) for nli-pooling.

A SIAMESE distilbert bi-encoder encodes the premise and hypothesis separately;
each sentence's token hidden states are POOLED into a single sentence vector,
then a fixed interaction and linear classifier produce the
3-way label. You control ONLY how the token states are pooled into a sentence
vector.

Implement:

    def build_pooling() -> dict:
        return {"pooling": ...}

Options:
  pooling : "sum"  -> attention-masked sum over token hidden states.
                      It does not divide by the number of non-padding tokens.
                      Padding positions never contribute.
            "cls"  -> the first-token ([CLS]) hidden state.
            "max"  -> attention-masked max over tokens.
            "mean" -> attention-masked arithmetic mean over non-padding tokens.
                      Every mode returns one fixed-width sentence vector.

All pooling modes share the same encoder, interaction, classifier, complete
corpus, optimizer budget, and evaluation domains. The verifier measures their
effect without publishing a preferred pooling operation in the editable
workspace.

Notes:
  * One deterministic three-epoch full-corpus run is followed by three evaluations.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your pooling configuration below
# ================================================================
def build_pooling() -> dict:
    # Native configuration; replace it to study another supported pooler.
    return {"pooling": "sum"}
# ================================================================
# END EDITABLE REGION
# ================================================================
