"""Strong baseline: SUCCESSIVE HALVING — cheap screen, then re-evaluate survivors on
MORE dev examples (Jamieson & Talwalkar; UCB-style allocation).

Round 1: screen all candidates on a small dev slice. Keep the top half. Round 2+:
re-evaluate the survivors on a LARGER dev slice (variance shrinks), halving each
round until one remains. The finalists get a stable, low-variance dev estimate, so
the chosen candidate generalizes to the disjoint held-out TEST set — clearly beating
the tiny-slice / pick-first strategies under the SAME execution budget.
Reference: vendor/prompt-optimization-lab/baselines/strategy_halving.py
"""


def select(candidates, ctx) -> str:
    dev = ctx["dev"]
    budget = ctx["budget"]
    ev = ctx["eval_on_dev"]
    alive = list(candidates)
    used = 0
    n_dev = len(dev)
    # Start with a small slice; grow it as the field halves. Budget-aware: stop
    # growing when the next round would exceed the remaining budget.
    slice_n = max(4, budget // (4 * max(1, len(candidates))))
    while len(alive) > 1:
        slice_n = min(slice_n, n_dev)
        # Would this round fit in the remaining budget? (upper bound: each survivor
        # is re-scored on `slice_n` fresh examples).
        rows = dev[:slice_n]
        scored = []
        for c in alive:
            a = ev(c, rows)
            scored.append((a, c))
            used += slice_n
        scored.sort(key=lambda t: t[0], reverse=True)
        keep = max(1, len(alive) // 2)
        alive = [c for _, c in scored[:keep]]
        slice_n = min(n_dev, slice_n * 2)   # survivors get more dev examples
        # If the next round can't afford re-evaluating survivors, stop with the top.
        if len(alive) > 1 and used + len(alive) * slice_n > budget:
            # spend the rest confirming the current top-2 on the largest affordable slice
            afford = max(4, (budget - used) // max(1, len(alive)))
            rows = dev[:min(n_dev, afford)]
            scored = [(ev(c, rows), c) for c in alive]
            scored.sort(key=lambda t: t[0], reverse=True)
            return scored[0][1]
    return alive[0]
