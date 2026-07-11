"""Summarization length-control surface (agent-editable).

FROZEN domain-matched summarizers decode THREE FIXED domain settings (xsum / cnndm
/ samsum) with beam search + no-repeat-3gram FIXED; you control ONLY the LENGTH
window of the decode, applied to all three settings. The summaries use mean
per-example ROUGE-L F1 (higher is better, gmean over 3 settings) against FIXED
references.

Implement:

    def build_length_config() -> dict:
        return {"min_length": ..., "max_length": ..., "length_penalty": ...}

The three knobs (transformers `model.generate`):
  min_length     : integer in [0, 200], no larger than max_length.
                   It is the minimum number of generated tokens.
                   Its measured effect is left to the benchmark.
  max_length     : integer in [1, 200], no smaller than min_length.
  length_penalty : finite in (0, 10]; >1.0 favours longer sequences,
                   <1.0 favours shorter.

Background:
  Length controls interact with each domain's source and target distribution. The
  three settings span 1-sentence (XSum), multi-sentence (CNN/DM), and dialogue
  (SAMSum) references, so a single window that is too short truncates the longer
  targets (recall dies) and one that is too long over-generates on the short
  targets (precision dies under F1). You must find a window that works across the
  three domains. No preferred window or measured ordering is published here.
  The native configuration remains runnable for no-edit verification.

Notes:
  * Inference-only. Deterministic. Runs the complete official test splits serially on one GPU.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your length-control decode config below
# ================================================================
def build_length_config() -> dict:
    # Native no-edit configuration; replace this mapping to test another window.
    return {"min_length": 1, "max_length": 20, "length_penalty": 0.2}
# ================================================================
# END EDITABLE REGION
# ================================================================
