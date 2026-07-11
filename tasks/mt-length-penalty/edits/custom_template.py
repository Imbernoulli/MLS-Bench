"""Machine-translation length-normalization surface (agent-editable).

A FROZEN OPUS-MT MarianMT model translates each complete pinned OPUS-100
source-to-English test split with a FIXED beam width
(num_beams 5); you control ONLY the LENGTH-NORMALIZATION policy. The translations
are scored on corpus sacreBLEU (higher is better) against FIXED English
references.

Implement:

    def build_length_config() -> dict:
        return {"length_penalty": 1.0, "min_length": 0, "max_new_tokens": 128}

The knobs (transformers `model.generate`):
  length_penalty : HF divides each beam's score by length**length_penalty.
                   length_penalty > 1.0 promotes LONGER sequences; < 1.0 promotes
                   SHORTER; 1.0 is plain mean-log-prob normalization. There is an
                   OPTIMUM (Wu et al. 2016 GNMT length penalty): too large
                   over-generates (this MarianMT model runs long by default -> the
                   brevity penalty is fine but n-gram precision drops), too small
                   over-truncates (brevity penalty). You must TUNE it to the
                   model + data; for this opus-mt-de-en on this set the sweet spot
                   is on the shorter side (around 0.6).
  min_length     : floor on generated length (0 == off). A small floor can stop
                   pathologically short outputs.
  max_new_tokens : cap on generated length (hard-capped at 160 by the harness).

Background:
  Beam search scores hypotheses by summed log-probability; the length_penalty
  (Wu et al. 2016, "Google's Neural Machine Translation System") controls the
  length bias. The DEFAULT here is a WEAK over-long length_penalty (2.0) that
  makes the model over-generate and drop BLEU; lowering it toward the ~0.6
  optimum tightens the output and recovers BLEU (over-shooting to 0.2 slightly
  over-truncates, so there is a genuine interior optimum to find).

Notes:
  * Inference-only. Deterministic. Evaluates each complete direction on one GPU; all directions contribute to the score.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your length-normalization decode config below
# ================================================================
def build_length_config() -> dict:
    # Default (weak): a strongly LONG-biased length penalty -> over-generation.
    return {"length_penalty": 2.0, "min_length": 0, "max_new_tokens": 128}
# ================================================================
# END EDITABLE REGION
# ================================================================
