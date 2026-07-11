"""Agent-editable SELECTION policy for codegen-best-of-n-select.

You control the candidate SELECTION / reranking policy. The harness draws a
FIXED pool of 8 candidates per problem (temperature 0.6, frozen) and calls
`select_candidate(candidates, problem, tok)`. You are given:
  * `candidates`: list of extracted candidate programs (strings),
  * `problem`: dict with `visible_tests`, `entry_point`, `prompt`, and
    `test_setup`; candidate structure and PROVIDED-test outcomes are available
    selection signals,
  * `tok`: the tokenizer.
Return the INDEX (int) of the candidate to submit.

You do NOT get the reserved tests. Scoring is on the DISJOINT RESERVED tests, so
selecting a candidate that overfits the provided assertion does not establish
general correctness. No selection signal is assumed to win before measurement.
"""
from __future__ import annotations

import common  # noqa: F401  (available on PYTHONPATH inside the harness)


# ================================================================
# EDITABLE REGION — return the chosen candidate index below
# ================================================================
def select_candidate(candidates, problem, tok):
    # Initial policy: choose the earliest candidate in the fixed pool.
    # Other pool-level signals may be used within this editable function.
    return 0
# ================================================================
# END EDITABLE REGION
# ================================================================
