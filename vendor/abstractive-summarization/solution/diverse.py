"""Summarization diverse-beam surface (agent-editable).

FROZEN domain-matched summarizers decode THREE FIXED domain settings (xsum / cnndm
/ samsum) with no-repeat-3gram + the per-domain length window FIXED; you control
ONLY beam grouping. Uses mean per-example ROUGE-L F1 (gmean over 3 settings).

Implement:

    def build_diverse_config() -> dict:
        return {"num_beams": ..., "num_beam_groups": ..., "diversity_penalty": ...}

  num_beams          : integer total beam width in [1, 12].
  num_beam_groups    : integer in [1, num_beams] that divides num_beams exactly;
                       1 is plain beam, >1 is diverse beam (Vijayakumar 2016).
  diversity_penalty  : exactly 0 when groups == 1; otherwise finite in (0, 10].

Background:
  Grouping changes interactions among candidate hypotheses while the verifier
  scores one committed summary. The measured effect depends on the full fixed
  protocol and is intentionally not stated here.
  Return a complete valid mapping; beam groups must divide the beam count. The
  native mapping remains runnable for no-edit verification.

Notes:
  * Inference-only. Deterministic. Runs the complete official test splits serially on one GPU.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your beam-grouping config below
# ================================================================
def build_diverse_config() -> dict:
    # Native no-edit configuration; replace this mapping to test another policy.
    return {"num_beams": 4, "num_beam_groups": 4, "diversity_penalty": 1.0}
# ================================================================
# END EDITABLE REGION
# ================================================================
