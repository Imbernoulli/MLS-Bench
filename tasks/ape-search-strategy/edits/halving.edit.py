"""Strong baseline: SUCCESSIVE HALVING — cheap screen, then re-evaluate survivors on
MORE dev examples (Jamieson & Talwalkar; UCB-style allocation).

Round 1: screen all candidates on a small dev slice. Keep the top half. Round 2+:
re-evaluate the survivors on a LARGER dev slice (variance shrinks), halving each round
until one remains. The finalists get a stable, low-variance dev estimate, so the
chosen candidate generalizes to the disjoint held-out TEST set — beating the
tiny-slice / pick-first strategies under the SAME execution budget.
Reference: vendor/prompt-optimization-lab/baselines/strategy_halving.py
"""

_FILE = "prompt-optimization-lab/solution/strategy.py"

_CONTENT = '''def select(candidates, ctx) -> str:
    dev = ctx["dev"]; budget = ctx["budget"]; ev = ctx["eval_on_dev"]
    alive = list(candidates); used = 0; n_dev = len(dev)
    slice_n = max(4, budget // (4 * max(1, len(candidates))))
    while len(alive) > 1:
        slice_n = min(slice_n, n_dev)
        rows = dev[:slice_n]
        scored = []
        for c in alive:
            a = ev(c, rows); scored.append((a, c)); used += slice_n
        scored.sort(key=lambda t: t[0], reverse=True)
        keep = max(1, len(alive) // 2)
        alive = [c for _, c in scored[:keep]]
        slice_n = min(n_dev, slice_n * 2)
        if len(alive) > 1 and used + len(alive) * slice_n > budget:
            afford = max(4, (budget - used) // max(1, len(alive)))
            rows = dev[:min(n_dev, afford)]
            scored = [(ev(c, rows), c) for c in alive]
            scored.sort(key=lambda t: t[0], reverse=True)
            return scored[0][1]
    return alive[0]'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 45, "end_line": 47, "content": _CONTENT},
]
