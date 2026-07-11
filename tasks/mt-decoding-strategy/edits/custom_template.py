"""Machine-translation decode-STRATEGY surface (agent-editable; monotonicity task).

For each complete pinned OPUS-100 source-to-English test split, you
choose HOW each translation is produced. The output is scored on corpus sacreBLEU
(higher is better) against FIXED English references.

Implement:

    def build_strategy() -> str:
        return "beam"

Options:
  "beam"        : decode the FROZEN opus-mt-de-en with a tuned config (beam 5,
                  length_penalty 1.0) — the strong, real translation.
  "greedy"      : decode the FROZEN model greedily (beam 1) — real but weaker.
  "copy_source" : DEGENERATE — return the German SOURCE unchanged. Wrong language
                  vs the English references -> ~0 BLEU.
  "first_token" : DEGENERATE — return only the first source word -> ~0 BLEU.
  "empty"       : DEGENERATE — return an empty string -> 0 BLEU.

Background:
  This task verifies the sacreBLEU metric is MONOTONE and UN-GAMEABLE. BLEU is a
  geometric mean of n-gram precisions with a brevity penalty, so copying the
  source German (a different language from the English references) or emitting
  empty/constant text scores ~0, while a real model decode scores clearly higher.
  You should pick the strategy that actually maximizes BLEU.

Notes:
  * Inference-only. Deterministic. Evaluates each complete direction on one GPU; all directions contribute to the score.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your decode strategy below
# ================================================================
def build_strategy() -> str:
    # Default (degenerate): copy the German source unchanged (~0 BLEU).
    return "copy_source"
# ================================================================
# END EDITABLE REGION
# ================================================================
