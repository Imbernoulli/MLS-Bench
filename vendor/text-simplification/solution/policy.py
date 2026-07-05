"""Text-simplification rewrite-POLICY surface (agent-editable; monotonicity task).

Across THREE FIXED simplification test settings (asset / turk / wiki), you choose
HOW each sentence is rewritten. The output is scored on corpus SARI (higher is
better) against each setting's FIXED multi-reference set; the task score is the
geometric mean over the three settings, so a policy must win on ALL THREE.

Implement:

    def build_policy() -> str:
        return "beam"

Options:
  "beam"        : simplify the FROZEN t5-base with a tuned config (beam 5,
                  no-repeat-3gram) — the strong, real simplification (SOTA-scale).
  "greedy"      : simplify the FROZEN model greedily (beam 1) — real but weaker.
  "truncate"    : keep the first 75% of the words (naive tail deletion).
  "first_token" : DEGENERATE FLOOR — return only the first source word. Low SARI.
  "empty"       : DEGENERATE FLOOR — return an empty string. Low SARI.

Background:
  This task verifies SARI is MONOTONE and UN-GAMEABLE. SARI compares the SOURCE,
  the system output, AND multiple references, rewarding correct ADD / KEEP / DELETE
  n-gram edits. A meaning-destroying output (empty / first-token) scores a
  genuinely LOW SARI on every setting, while a real T5 simplifier reaches the
  SOTA-scale top. You should pick the policy that actually maximizes SARI.

Notes:
  * Inference-only. Deterministic. Runs on a single GPU in a few minutes.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return your rewrite policy below
# ================================================================
def build_policy() -> str:
    # Default (degenerate floor): emit nothing (no simplification -> low SARI).
    return "empty"
# ================================================================
# END EDITABLE REGION
# ================================================================
