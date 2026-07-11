"""Machine-translation diverse-beam surface (agent-editable).

A FROZEN OPUS-MT model translates each complete pinned OPUS-100 source-to-English test split with DIVERSE beam
search (Vijayakumar et al. 2016); num_beams is FIXED at 8. You control the number
of beam GROUPS and the diversity penalty. Scored on corpus sacreBLEU (higher).

Implement:

    def build_divbeam_config() -> dict:
        return {"num_beam_groups": 1, "diversity_penalty": 0.0}

  num_beam_groups   : partition the 8 beams into this many groups (1 = plain beam;
                      must divide 8 -> snapped to 1/2/4/8).
  diversity_penalty : Hamming penalty added across groups. It must be zero for
                      one group and strictly positive for multiple groups.

Background:
  Diverse beam search improves the DIVERSITY of the n-best list, but for
  SINGLE-BEST MT quality it is a trade-off: modest grouping (2 groups, small
  penalty) stays near plain beam, while many groups + a large penalty (8 groups,
  >=1.0) push the top hypothesis off the high-probability path -> lower BLEU.
  Plain beam (1 group) is the reference; over-diversified is the degenerate.

Notes:
  * Inference-only. Deterministic. Aggregated over three directions. Minute-scale.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your diverse-beam config below
# ================================================================
def build_divbeam_config() -> dict:
    # Default (weak): 8 groups + a large diversity penalty -> off the MAP path.
    return {"num_beam_groups": 8, "diversity_penalty": 1.5}
# ================================================================
# END EDITABLE REGION
# ================================================================
