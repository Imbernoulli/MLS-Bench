"""Pair-encoding surface (agent-editable) for nli-pair-encoding.

A DistilBERT encoder and a 3-way NLI classifier are trained on the complete
labeled SNLI train split, then scored on three complete evaluation splits.
You select a complete CROSS-ENCODER or SIAMESE BI-ENCODER architecture.

Implement:

    def build_encoding() -> dict:
        return {"encoding": ...}

Options:
  encoding : "cross"   -> CROSS-ENCODER. Premise and hypothesis are concatenated
                          into ONE sequence ``[CLS] premise [SEP] hypothesis`` and
                          jointly attended; the [CLS] state is classified. Premise
                          and hypothesis tokens attend to each other.
             "siamese" -> SIAMESE bi-encoder. Premise and hypothesis use separate
                          64-token inputs to a shared encoder, mean pooling, and a
                          fixed InferSent readout over u and v. This is the standard
                          complete bi-encoder alternative, not an isolation of only
                          one internal primitive.

The two architectures place premise-hypothesis interaction at different stages.
They use the same pretrained revision, complete corpus, optimizer-step budget,
seed, and evaluation domains. The verifier measures their effect without
publishing a preferred architecture in the editable workspace. Unsupported
selectors fail before model construction; no alternative is substituted on error.
All metrics come from the live trained model.

Notes:
  * One deterministic three-epoch full-corpus run is followed by three evaluations.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your pair-encoding configuration below
# ================================================================
def build_encoding() -> dict:
    # Native configuration; replace it to study the other supported architecture.
    return {"encoding": "siamese"}
# ================================================================
# END EDITABLE REGION
# ================================================================
