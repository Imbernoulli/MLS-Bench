"""Machine-translation decoding-temperature surface (agent-editable).

A FROZEN OPUS-MT model translates each complete pinned OPUS-100 source-to-English test split in SAMPLING mode
(do_sample, top_p 0.95, seed fixed); you control ONLY the softmax `temperature`.
Scored on corpus sacreBLEU (higher is better).

Implement:

    def build_temperature() -> float:
        return 0.3

  temperature : scales logits before softmax. HIGH (>=1.0) flattens the
                distribution -> more random, lower-BLEU samples. LOW (->0)
                sharpens toward the argmax (approaching greedy) -> higher BLEU for
                a peaked MT model. There is a monotone benefit to LOWERING it here
                (MT is not open-ended generation); the sweet spot is low
                (~0.3-0.5). Clamped to [0.05, 5.0].

Notes:
  * Inference-only. RNG seed fixed for reproducibility. Aggregated over three
    directions. Minute-scale.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your sampling temperature below
# ================================================================
def build_temperature() -> float:
    # Default (weak): a high temperature -> noisy, low-BLEU samples.
    return 1.5
# ================================================================
# END EDITABLE REGION
# ================================================================
