"""Encoder update policy surface for nli-finetune.

The fixed runner trains a transformer cross-encoder for 3-way natural language
inference on the complete labeled SNLI training split. Return one of:
  - {"encoder": "frozen"}: update only the classifier head.
  - {"encoder": "finetune"}: update the classifier head and encoder.
"""
from __future__ import annotations


def build_finetune() -> dict:
    """Select the encoder update policy for the fixed NLI trainer."""
    # Native configuration; replace it to study another supported policy.
    return {"encoder": "frozen"}
