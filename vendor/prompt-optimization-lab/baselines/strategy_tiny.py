"""Weak baseline: TINY-SLICE — judge EVERY candidate on the same tiny dev slice.

Evaluate all candidates on the first few dev examples and take the argmax. This
spends the budget "fairly" but each estimate is high-variance, so it overfits dev
noise and frequently picks a candidate that does not generalize to the held-out TEST
set (ties are common on a tiny slice and resolve arbitrarily).
Reference: vendor/prompt-optimization-lab/baselines/strategy_tiny.py
"""


def select(candidates, ctx) -> str:
    dev = ctx["dev"]
    budget = ctx["budget"]
    ev = ctx["eval_on_dev"]
    # Split the budget equally across ALL candidates -> a tiny per-candidate slice.
    per = max(2, budget // max(1, len(candidates)))
    slice_rows = dev[:per]
    best, best_acc = candidates[0], -1.0
    for c in candidates:
        a = ev(c, slice_rows)
        if a > best_acc:
            best_acc, best = a, c
    return best
