"""Weak baseline for ape-search-algorithm (Instruction SEARCH ALGORITHM (iterative / beam refinement under a dev budget)).

Reference: pasted into solution/searchalgo.py via the edit op.
"""

import common  # noqa: F401


def search(ctx):
    # Weak: propose exactly ONE candidate and return it blindly — no dev evaluation,
    # no refinement; generalizes only by luck.
    cands = ctx["induce"](1)
    return cands[0] if cands else ""
