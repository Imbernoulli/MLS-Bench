"""Text-simplification MODEL-CAPACITY / CHECKPOINT-CHOICE surface (agent-editable).

You control ONLY WHICH FROZEN, staged-offline pretrained simplifier decodes a FIXED
small ASSET/TURK/WikiAuto test slice under an IDENTICAL strong beam decode config
(num_beams=5, no_repeat_ngram_size=3, length_penalty=1.0, all FIXED for every
choice). The rewrites are scored on corpus SARI (higher is better) against the
FIXED multi-reference set.

Implement:

    def build_model_choice() -> str:
        return "base_turk"

Must be one of the three staged checkpoints (all from the wiki_auto_asset_turk
family of simplification fine-tunes; NONE trained here — this is a "which existing
checkpoint" lever, not a training task):
  "small_turk"     : t5-small-finetuned-turk-text-simplification (t5-small, 60M
                     params, mainly TurkCorpus-style lexical edits).
  "small_wikiauto" : t5-small-finetuned-text-simplification (t5-small, 60M params,
                     broader wiki_auto_asset_turk fine-tune mix).
  "base_turk"      : t5-base-finetuned-turk-text-simplification (t5-base, ~220M
                     params — 3.7x the parameters of either t5-small checkpoint;
                     the model used by every other simp-* task).

Background:
  Holding the decode config fixed, MORE MODEL CAPACITY (t5-base vs t5-small) is the
  standard "bigger backbone helps" lever in seq2seq NLP; measured on GPU (k1 H20,
  2026-07-05), SARI improves `small_wikiauto` (39.9/39.1/38.2) < `small_turk`
  (41.9/42.0/42.2) < `base_turk` (45.1/43.7/43.3) on all three settings -- capacity
  AND fine-tuning-data family both matter, and base_turk (the model every other
  simp-* task uses) is the clean strongest choice. The DEFAULT here is the WEAKEST
  measured checkpoint (`small_wikiauto`).

Notes:
  * Inference-only. Deterministic. Loads a DIFFERENT frozen checkpoint per choice
    (not the one loaded by the shared harness default) but runs on a single GPU in
    well under a minute either way.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your model choice below
# ================================================================
def build_model_choice() -> str:
    # Default (weak): smaller t5-small checkpoint, broader (less-targeted) fine-tune.
    return "small_wikiauto"
# ================================================================
# END EDITABLE REGION
# ================================================================
