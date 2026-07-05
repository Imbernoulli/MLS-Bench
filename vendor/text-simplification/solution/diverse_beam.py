"""Text-simplification DIVERSE BEAM SEARCH surface (agent-editable).

A FROZEN pretrained t5-base simplifier rewrites a FIXED small ASSET/TURK/WikiAuto
test slice with beam search (num_beams=6, no_repeat_ngram_size=3 both FIXED); you
control ONLY the GROUP BEAM SEARCH config (number of groups + diversity penalty
between groups). The rewrites are scored on corpus SARI (higher is better; only the
TOP-1 hypothesis is scored) against the FIXED multi-reference set.

Implement:

    def build_diverse_beam_config() -> dict:
        return {"num_beam_groups": 1, "diversity_penalty": 0.0}

The two knobs (transformers `model.generate`, Vijayakumar et al. 2016):
  num_beam_groups   : split the 6 beams into this many groups (must divide 6: 1,
                       2, 3, or 6). 1 = plain beam search (no grouping).
  diversity_penalty : penalty applied to a later group's tokens for matching an
                       earlier group's tokens at the same step (only has effect
                       when num_beam_groups > 1). Hard-capped at 5.0.

Background:
  Diverse beam search trades single-best (top-1) sequence quality for hypothesis
  diversity across groups — useful when you want several DIFFERENT good outputs,
  but for a task that is scored on only the TOP-1 decode (as here), pushing beams
  apart with a high diversity penalty makes the top-1 hypothesis WORSE, not better:
  it is diverted away from the single highest-probability sequence a plain
  (ungrouped) beam search would find. The DEFAULT here is a WEAK config with many
  small groups and a strong diversity penalty.

Notes:
  * Inference-only. Deterministic. Runs on a single GPU in well under a minute.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your diverse-beam group config below
# ================================================================
def build_diverse_beam_config() -> dict:
    # Default (weak): many groups, strong diversity penalty -> top-1 hurt.
    return {"num_beam_groups": 6, "diversity_penalty": 3.0}
# ================================================================
# END EDITABLE REGION
# ================================================================
