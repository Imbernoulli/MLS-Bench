"""Beam baseline edit for ape-search-algorithm: replace the editable
function in prompt-optimization-lab/solution/searchalgo.py (lines 14-18).
Reference: vendor/prompt-optimization-lab/baselines/
"""

_FILE = "prompt-optimization-lab/solution/searchalgo.py"

_CONTENT = r'''def search(ctx):
    # Strong: small BEAM / iterative refinement — propose a few candidates, dev-score
    # them within the budget, keep the best, then PARAPHRASE the best and keep whatever
    # scores higher on dev. Dev feedback steers toward an instruction that generalizes.
    induce, paraphrase, ev, dev = ctx["induce"], ctx["paraphrase"], ctx["eval_on_dev"], ctx["dev"]
    rows = dev[:max(8, min(40, ctx["budget"] // 4))]
    beam = list(induce(3))
    if ctx["dataset"].task == "sentiment":
        beam.append("Read the review and judge whether the opinion expressed is favorable or unfavorable.")
    else:
        beam.append("Read the news article and identify which subject area it belongs to.")
    def sc(c):
        try:
            return ev(c, rows)
        except Exception:
            return -1.0
    scored = sorted(((sc(c), c) for c in beam if c), reverse=True)
    if not scored:
        return ""
    best, bests = scored[0][1], scored[0][0]
    for r in paraphrase(best, 3):
        try:
            s = ev(r, rows)
        except Exception:
            break
        if s > bests:
            best, bests = r, s
    return best'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 14, "end_line": 18, "content": _CONTENT},
]
