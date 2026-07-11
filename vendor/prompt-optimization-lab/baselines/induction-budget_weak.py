"""Weak baseline for ape-induction-budget (Shared PROPOSAL-vs-EVALUATION BUDGET ALLOCATION).

Reference: pasted into solution/budget.py via the edit op.
"""

import common  # noqa: F401


def allocate(ctx):
    # Weak: spend the ENTIRE shared budget proposing candidates; nothing remains for
    # dev evaluation, so the choice is blind (first proposal) and rarely generalizes.
    cands = ctx["propose"](ctx["budget"])
    return cands[0] if cands else ""
