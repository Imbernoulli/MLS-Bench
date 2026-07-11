"""Summarization nucleus (top-p) cutoff surface (agent-editable).

FROZEN domain-matched summarizers decode THREE FIXED domain settings (xsum / cnndm
/ samsum) with sampling ON (temperature=1.0) + no-repeat-3gram + the per-domain
length window FIXED; you control ONLY the nucleus cutoff top_p. Scored on corpus
ROUGE-L F1 (gmean over the 3 settings).

Implement:

    def build_top_p() -> float:
        return ...

  top_p : nucleus mass in the documented probability interval. Sampling draws from
          the smallest token set whose cumulative probability reaches that mass.
          Invalid or non-finite values fail instead of being clamped. The measured
          relationship to ROUGE is intentionally omitted from this interface.

Background:
  Compare valid nucleus masses on the fixed multi-domain protocol.
  The native value remains runnable for no-edit verification.

Notes:
  * Inference-only. Sampling is seeded (seed 42). Runs the complete official test splits serially on one GPU.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your nucleus (top-p) cutoff below
# ================================================================
def build_top_p() -> float:
    # Native no-edit value; replace it to test another nucleus mass.
    return 1.0
# ================================================================
# END EDITABLE REGION
# ================================================================
