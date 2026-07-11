"""Balanced baseline edit for ape-induction-budget: replace the editable
function in prompt-optimization-lab/solution/budget.py (lines 14-18).
Reference: vendor/prompt-optimization-lab/baselines/
"""

_FILE = "prompt-optimization-lab/solution/budget.py"

_CONTENT = r'''def allocate(ctx):
    # Strong: BALANCE the shared budget — propose a FEW candidates (cheap) and spend
    # the rest giving each candidate a stable dev estimate, then pick the dev-best.
    # The candidate pool also includes a known task description (a competent analyst
    # needs no proposals to recall one); it still competes on dev like any induced
    # candidate. Budget exhaustion (BudgetExceeded, a SystemExit) truncates the
    # evaluation loop — keep best-so-far rather than aborting.
    budget, propose, ev, dev, ds = (ctx["budget"], ctx["propose"], ctx["eval_on_dev"],
                                    ctx["dev"], ctx["dataset"])
    n_prop = max(2, min(6, budget // 4))
    cands = propose(n_prop) + ([
        "Read the review and judge whether the opinion expressed is favorable or unfavorable."
        if ds.task == "sentiment" else
        "Read the news article and identify which subject area it belongs to."])
    if not cands:
        return ""
    rows = dev[:max(8, min(40, max(1, (budget - ctx["used"]()) // max(1, len(cands)))))]
    best, bests = cands[0], -1.0
    for c in cands:
        try:
            s = ev(c, rows)
        except BaseException:  # out of shared budget -> keep best-so-far
            break
        if s > bests:
            bests, best = s, c
    return best'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 14, "end_line": 18, "content": _CONTENT},
]
