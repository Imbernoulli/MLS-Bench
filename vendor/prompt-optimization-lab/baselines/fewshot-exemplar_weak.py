"""Weak baseline for ape-fewshot-exemplar (Few-shot EXEMPLAR selection for reverse-mode induction).

Reference: pasted into solution/exemplar.py via the edit op.
"""

import common  # noqa: F401


def select_exemplars(pool, ctx):
    # Weak: a single arbitrary exemplar — narrow, noisy induction that rarely yields
    # a generalizing instruction.
    return [pool[0]]
