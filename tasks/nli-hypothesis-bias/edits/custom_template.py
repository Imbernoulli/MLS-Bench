"""Hypothesis-bias surface (agent-editable) for nli-hypothesis-bias.

A 3-way NLI classifier is trained on the complete labeled SNLI split and scored on
3-way accuracy over three evaluation domains (SNLI test, MNLI matched dev, MNLI
mismatched dev). The two arms use the same cross-encoder architecture and differ
only in whether the premise tokens are supplied or replaced by an empty sequence.
You control
ONLY whether the model is allowed to use the premise.

Implement:

    def build_bias() -> dict:
        return {"use_premise": ...}

Options:
  use_premise : True  -> the cross-encoder receives premise and hypothesis tokens.
                         Both sequences share one joint attention stack.
                         All other training controls remain fixed.
                False -> the same cross-encoder receives an empty premise and the
                         original hypothesis tokens.
                         All other training controls remain fixed.

This isolates premise availability without changing model class, corpus, update
count, optimizer, or evaluation domains. Hypothesis-only behavior is a standard
NLI diagnostic (Gururangan et al., NAACL 2018). The verifier measures both
supported policies without publishing their ordering in the editable workspace.
No alternate policy is substituted after a failure.

Notes:
  * One deterministic three-epoch full-corpus run is followed by three evaluations.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your bias-mitigation configuration below
# ================================================================
def build_bias() -> dict:
    # Native configuration; replace it to study the other supported policy.
    return {"use_premise": False}
# ================================================================
# END EDITABLE REGION
# ================================================================
