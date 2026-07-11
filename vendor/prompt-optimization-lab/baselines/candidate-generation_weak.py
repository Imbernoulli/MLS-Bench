"""Weak baseline for ape-candidate-generation (Candidate GENERATION (the APE proposer)).

Reference: pasted into solution/propose.py via the edit op.
"""

import common  # noqa: F401


def propose(ctx):
    # Weak: a single generic instruction — fixed dev-selection has nothing useful
    # to pick, so it selects a vague prompt that scores well below a genuine task description (MEASURED 2026-07-09: 0.6467/0.7533 test acc on agnews/sst2 vs strong 0.7767/0.9100).
    return ["Classify the text."]
