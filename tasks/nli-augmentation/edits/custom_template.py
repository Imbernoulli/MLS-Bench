"""Data-augmentation surface (agent-editable) for nli-augmentation.

A DistilBERT cross-encoder is trained on the complete labeled SNLI training
split with a fixed-size optional transformation; evaluation rows are unchanged. You
control ONLY the TRAIN-TIME DATA AUGMENTATION.

Implement:

    def build_augment() -> dict:
        return {"augment": ...}

Options:
  augment : "none"     -> no augmentation.
            "swap"     -> reverse premise/hypothesis on contradiction rows while
                          preserving the contradiction label.
            "negation" -> apply heuristic lexical negation and relabel the row as
                          contradiction; this diagnostic can introduce label noise.
                          Every arm retains the same number of training rows.

The transformation is applied only to training data. Corpus order, update count,
model, optimizer, and complete evaluation domains remain fixed. The verifier
measures each supported policy without publishing a preferred configuration in
the editable workspace.

Notes:
  * One deterministic three-epoch full-corpus run is followed by three evaluations.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your augmentation configuration below
# ================================================================
def build_augment() -> dict:
    # Native configuration; replace it to study another supported policy.
    return {"augment": "none"}
# ================================================================
# END EDITABLE REGION
# ================================================================
