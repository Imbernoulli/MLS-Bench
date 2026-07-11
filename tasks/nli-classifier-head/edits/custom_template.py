"""Classifier-head surface (agent-editable) for nli-classifier-head.

A siamese DistilBERT bi-encoder with a fixed interaction over
mean-pooled sentence vectors produces the feature the 3-way classifier sees. You
control ONLY the CLASSIFIER-HEAD DEPTH.

Implement:

    def build_classifier() -> dict:
        return {"head": ...}

Options:
  head : "linear" -> a single affine projection over the interaction features.
                     It has no intermediate hidden representation.
         "mlp"    -> a two-layer head (Linear -> GELU -> Dropout -> Linear) with
                     a fixed hidden width.

Both heads receive identical interaction features and share the complete corpus,
encoder, optimizer budget, and evaluator. Their measured effect is verifier-owned;
no preferred head is stated in the editable workspace.

Notes:
  * One deterministic three-epoch full-corpus run is followed by three evaluations.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your classifier-head configuration below
# ================================================================
def build_classifier() -> dict:
    # Native configuration; replace it to study another supported head.
    return {"head": "linear"}
# ================================================================
# END EDITABLE REGION
# ================================================================
