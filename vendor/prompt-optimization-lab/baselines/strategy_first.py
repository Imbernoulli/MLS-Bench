"""Weak baseline: PICK-FIRST — return candidates[0], spending no dev budget.

No search at all: commit to the first candidate. With a mixed candidate pool this
often lands on a mediocre (or misleading) instruction and generalizes poorly to the
held-out TEST set. Reference: vendor/prompt-optimization-lab/solution/strategy.py.
"""


def select(candidates, ctx) -> str:
    return candidates[0]
