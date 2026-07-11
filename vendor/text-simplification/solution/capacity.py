"""Text-simplification MODEL-CAPACITY / CHECKPOINT-CHOICE surface (agent-editable).

You control ONLY WHICH FROZEN, staged-offline pretrained simplifier decodes the
complete pinned official ASSET, TurkCorpus, and WikiAuto test partitions under
one decode configuration that is identical for every choice. The rewrites are
scored on corpus SARI (higher is better) against the
FIXED multi-reference set.

Implement:

    def build_model_choice() -> str:
        return ...

Must be one of the three staged checkpoints (all from the wiki_auto_asset_turk
family of simplification fine-tunes; NONE trained here — this is a "which existing
checkpoint" lever, not a training task):
  "small_turk"     : a staged T5-small checkpoint.
  "small_wikiauto" : another staged T5-small checkpoint.
  "base_turk"      : a staged T5-base checkpoint.
  Every choice is loaded offline with the same decode settings.
  Invalid or missing checkpoint selectors fail before generation.
  No checkpoint is substituted after a load failure.
  No checkpoint ordering is prescribed.

Background:
  Checkpoint size and fine-tuning data can affect simplification behavior.
  This task compares the staged candidates under one fixed evaluation protocol.
  Exact metrics, baseline ordering, and preferred checkpoints are not exposed.
  Every declared setting must produce complete predictions and finite SARI.
  The native selector remains runnable for no-edit verification.
  Use submitted verifier results to compare alternatives.
  Runtime failures abort verification.

Notes:
  * Inference-only and deterministic. Loads a DIFFERENT frozen checkpoint per
    choice and must generate a complete prediction for every official test example.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your model choice below
# ================================================================
def build_model_choice() -> str:
    # Native no-edit selector; replace it to test another staged checkpoint.
    return "small_wikiauto"
# ================================================================
# END EDITABLE REGION
# ================================================================
