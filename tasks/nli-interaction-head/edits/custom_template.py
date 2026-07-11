"""Interaction-head surface (agent-editable) for nli-interaction-head.

A SIAMESE distilbert bi-encoder mean-pools the premise to a vector u and the
hypothesis to a vector v (shared encoder, fine-tuned). You control ONLY how u and
v are COMBINED into the feature the linear 3-way classifier sees.

Implement:

    def build_head() -> dict:
        return {"interaction": ...}

Options:
  interaction : "concat"    -> the classifier sees only [u; v].
                "infersent" -> the classifier sees [u; v; |u - v|; u * v], the
                               InferSent matching features (Conneau et al., EMNLP
                               2017). The element-wise difference and product give
                               the head explicit alignment/contrast signals.

Both interaction vectors are deterministic functions of the same sentence
representations. Corpus, encoder, optimizer budget, classifier class, and
evaluation domains remain fixed. The verifier measures their effect without
publishing a preferred interaction in the editable workspace.

Notes:
  * One deterministic three-epoch full-corpus run is followed by three evaluations.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your interaction-head configuration below
# ================================================================
def build_head() -> dict:
    # Native configuration; replace it to study another supported interaction.
    return {"interaction": "concat"}
# ================================================================
# END EDITABLE REGION
# ================================================================
