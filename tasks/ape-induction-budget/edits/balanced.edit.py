"""Balanced baseline edit for ape-induction-budget: replace the editable
function in prompt-optimization-lab/solution/budget.py (lines 14-18).
Reference: vendor/prompt-optimization-lab/baselines/
"""

_FILE = "prompt-optimization-lab/solution/budget.py"

_CONTENT = r'''def allocate(ctx):
    # Strong: BALANCE the shared budget — propose a FEW candidates (cheap) and spend
    # the rest giving each a stable dev estimate, then pick the dev-best. Few proposals
    # + enough dev each beats propose-everything/blind-pick under the same budget.
    budget, propose, ev, dev = ctx["budget"], ctx["propose"], ctx["eval_on_dev"], ctx["dev"]
    n_prop = max(2, min(6, budget // 4))
    cands = propose(n_prop)
    if not cands:
        return ""
    rows = dev[:max(8, min(40, max(1, (budget - ctx["used"]()) // max(1, len(cands)))))]
    best, bests = cands[0], -1.0
    for c in cands:
        try:
            s = ev(c, rows)
        except Exception:
            break
        if s > bests:
            bests, best = s, c
    return best'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 14, "end_line": 18, "content": _CONTENT},
]
