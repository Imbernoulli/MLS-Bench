"""Allpropose baseline edit for ape-induction-budget: replace the editable
function in prompt-optimization-lab/solution/budget.py (lines 14-18).
Reference: vendor/prompt-optimization-lab/baselines/
"""

_FILE = "prompt-optimization-lab/solution/budget.py"

_CONTENT = r'''def allocate(ctx):
    # Weak: spend the ENTIRE shared budget proposing candidates; nothing remains for
    # dev evaluation, so the choice is blind (first proposal) and rarely generalizes.
    cands = ctx["propose"](ctx["budget"])
    return cands[0] if cands else ""'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 14, "end_line": 18, "content": _CONTENT},
]
