"""Summarization beam / repetition surface (agent-editable).

FROZEN domain-matched summarizers decode THREE FIXED domain settings (xsum / cnndm
/ samsum) with a FIXED per-domain length window; you control ONLY the BEAM SEARCH
and REPETITION control. The summaries use mean per-example ROUGE-L F1 (higher is
better, gmean over the 3 settings) against FIXED references.

Implement:

    def build_beam_config() -> dict:
        return {"num_beams": ..., "no_repeat_ngram_size": ..., "repetition_penalty": ...}

The three knobs (transformers `model.generate`):
  num_beams            : beam width, represented as an integer in [1, 12].
                         Different widths change the search procedure and cost;
                         the benchmark does not publish their measured ordering
                         in this agent-visible file.
  no_repeat_ngram_size : integer in [0, 20] selecting the forbidden n-gram size.
                         Zero disables it. Nonzero values forbid repeated spans
                         of the selected size. Compare them under the fixed
                         multi-domain evaluation.
  repetition_penalty   : finite in (0, 10]; >1 discourages repeats; 1 is off.

Background:
  These controls interact with summary length and domain. Select a complete
  configuration empirically; no baseline ranking or measured value is exposed
  here. The native configuration remains runnable for no-edit verification.

Notes:
  * Inference-only. Deterministic. Runs the complete official test splits serially on one GPU.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your beam / repetition decode config below
# ================================================================
def build_beam_config() -> dict:
    # Native no-edit configuration; replace this mapping to test another policy.
    return {"num_beams": 1, "no_repeat_ngram_size": 0, "repetition_penalty": 1.0}
# ================================================================
# END EDITABLE REGION
# ================================================================
