"""Class-weighting surface (agent-editable) for nli-class-weighting.

A DistilBERT cross-encoder is trained on every labeled SNLI training example.
The observed label frequencies are authenticated from that complete split, and
the complete evaluation domains are never resampled. You control only the loss
class-weighting policy.

Implement:

    def build_weighting() -> dict:
        return {"weights": [entailment, neutral, contradiction]}

Return exactly three finite numeric weights in canonical label order:
`entailment`, `neutral`, `contradiction`. Each value must be in [0.25, 2.0]
and their arithmetic mean must be one. The constraint keeps the overall loss
scale fixed while allowing a meaningful cost-sensitive objective.

Every choice uses identical examples, model, optimizer-step inventory, and
evaluation data. Its measured effect is verifier-owned; no preferred policy is
stated in the editable workspace.

Notes:
  * One deterministic three-epoch full-corpus run is followed by three evaluations.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your class-weighting configuration below
# ================================================================
def build_weighting() -> dict:
    # Canonical order: entailment, neutral, contradiction.
    return {"weights": [1.0, 1.0, 1.0]}
# ================================================================
# END EDITABLE REGION
# ================================================================
