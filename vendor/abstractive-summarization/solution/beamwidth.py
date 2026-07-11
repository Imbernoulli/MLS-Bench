"""Summarization beam-WIDTH surface (agent-editable).

FROZEN domain-matched summarizers decode THREE FIXED domain settings (xsum / cnndm
/ samsum) with a FIXED per-domain length window and a FIXED no-repeat-3gram block;
you control ONLY the beam WIDTH. Uses mean per-example ROUGE-L F1 (gmean over 3
settings).

Implement:

    def build_beam_width() -> int:
        return ...

  num_beams : integer in [1, 12] controlling search width. Different
              values change both decoding behavior and compute. Their measured
              ordering is intentionally omitted from the agent-visible
              interface.

Background:
  Evaluate candidate widths on the fixed multi-domain protocol.
  The native value is retained solely as a runnable no-edit policy.

Notes:
  * Inference-only. Deterministic. Runs the complete official test splits serially on one GPU.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your beam width below
# ================================================================
def build_beam_width() -> int:
    # Native no-edit value; replace it to test another width.
    return 1
# ================================================================
# END EDITABLE REGION
# ================================================================
