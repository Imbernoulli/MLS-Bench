"""Text-simplification rewrite-POLICY surface (agent-editable).

Across THREE FIXED simplification test settings (asset / turk / wiki), you choose
HOW each sentence is rewritten. The output is scored on corpus SARI (higher is
better) against each setting's FIXED multi-reference set; the task score is the
geometric mean over the three settings, so a policy must win on ALL THREE.

Options:
  "beam"        : run the FROZEN t5-base simplifier with a fixed multi-beam config
                  and no-repeat n-gram blocking.
  "greedy"      : run the FROZEN t5-base simplifier with greedy decoding.
  "truncate"    : keep the first 75% of the words (naive tail deletion).
  "first_token" : return only the first source word.
  "empty"       : return an empty string.

Background:
  SARI compares the SOURCE, the system output, AND multiple references, rewarding
  correct ADD / KEEP / DELETE n-gram edits. The terms trade off across the three
  datasets. No policy ordering is prescribed; compare submitted verifier results.

Notes:
  * Inference-only and deterministic. Model-backed policies must generate a
    complete prediction for every official test example; shortcut policies do not
    constitute execution evidence for the frozen model path.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your rewrite policy below
# ================================================================
def build_policy() -> str:
    # Native no-edit selector; replace it to test another supported policy.
    return "empty"
# ================================================================
# END EDITABLE REGION
# ================================================================
