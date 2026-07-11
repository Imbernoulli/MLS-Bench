"""Oneshot baseline edit for ape-search-algorithm: replace the editable
function in prompt-optimization-lab/solution/searchalgo.py (lines 14-18).
Reference: vendor/prompt-optimization-lab/baselines/
"""

_FILE = "prompt-optimization-lab/solution/searchalgo.py"

_CONTENT = r'''def search(ctx):
    # Weak: propose exactly ONE candidate and return it blindly — no dev evaluation,
    # no refinement; generalizes only by luck.
    cands = ctx["induce"](1)
    return cands[0] if cands else ""'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 14, "end_line": 18, "content": _CONTENT},
]
