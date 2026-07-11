"""Summarization post-truncation surface (agent-editable).

FROZEN domain-matched summarizers decode THREE FIXED domain settings (xsum / cnndm
/ samsum) with one fixed decode config identical for every candidate;
you control ONLY a pure POST-PROCESS: how many leading SENTENCES of the decoded
summary to KEEP. Uses mean per-example ROUGE-L F1 (gmean over 3 settings).

Implement:

    def build_keep_sentences() -> int:
        return ...

  keep_sentences : integer in [0, 10000] selecting leading sentences to retain.
                   A sufficiently large value retains every decoded sentence.
                   The benchmark measures the precision/recall tradeoff across
                   all three fixed domains.
                   No preferred value is disclosed here.

Background:
  Compare valid retention counts under the fixed decode and scoring protocol.
  Measured baseline ordering is not part of the solution interface.
  The native value remains runnable for no-edit verification.

Notes:
  * Inference-only. Deterministic. Runs the complete official test splits serially on one GPU.
"""
from __future__ import annotations


# ================================================================
# EDITABLE REGION — return how many leading sentences to keep
# ================================================================
def build_keep_sentences() -> int:
    # Native no-edit value; replace it to test another retention count.
    return 1
# ================================================================
# END EDITABLE REGION
# ================================================================
