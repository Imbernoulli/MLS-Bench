"""Machine-translation length-normalization surface (agent-editable).

A FROZEN OPUS-MT MarianMT model translates each complete pinned OPUS-100
source-to-English test split with a FIXED beam width
(num_beams 5); you control ONLY the LENGTH-NORMALIZATION policy. The translations
are scored on corpus sacreBLEU (higher is better) against FIXED English
references.

Implement:

    def build_length_config() -> dict:
        return {"length_penalty": 1.0}

The knobs (transformers `model.generate`):
  length_penalty : HF divides each beam's score by length**length_penalty.
                   Values above and below 1.0 change the relative preference for
                   longer and shorter hypotheses. The coefficient must be tuned
                   against the fixed corpus metric.

Background:
  Beam search scores hypotheses by sequence probability; the length penalty
  (Wu et al. 2016, "Google's Neural Machine Translation System") controls how
  that score is normalized by hypothesis length.

Notes:
  * Inference-only. Deterministic. Evaluates each complete direction on one GPU; all directions contribute to the score.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your length-normalization decode config below
# ================================================================
def build_length_config() -> dict:
    # Initial policy; select the coefficient justified by the fixed metric.
    return {"length_penalty": 2.0}
# ================================================================
# END EDITABLE REGION
# ================================================================
