"""Sequence-length / truncation surface (agent-editable) for nli-truncation.

A DistilBERT cross-encoder is trained on the complete labeled SNLI training
split. You control only the sequence-length cap for the joint
``[CLS] premise [SEP] hypothesis`` sequence.

Implement:

    def build_truncation() -> dict:
        return {"max_len": ...}

Options:
  max_len : an integer cap on the joint sequence length in [8, 128]. Invalid
            values fail rather than being clamped. Tokenization, padding, and
            truncation use the selected cap for training and every evaluation.
            All other model and optimizer controls remain fixed.

Every supported cap uses the complete training and evaluation row inventory and
the same number of optimizer updates. The verifier measures the effect without
publishing a preferred cap in the editable workspace.

Notes:
  * One deterministic three-epoch full-corpus run is followed by three evaluations.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your truncation configuration below
# ================================================================
def build_truncation() -> dict:
    # Native configuration; replace it to study another supported cap.
    return {"max_len": 16}
# ================================================================
# END EDITABLE REGION
# ================================================================
