"""Unmeasured candidate: pick the longest candidate.

A length heuristic with no assertion execution. It is a deterministic
content-only selector.
Reference surface: vendor/code-generation-lab/solution/policy_select.py
"""

_FILE = "code-generation-lab/solution/policy_select.py"

_CONTENT = '''def select_candidate(candidates, problem, tok):
    best_i, best_len = 0, -1
    for i, c in enumerate(candidates):
        L = len((c or "").strip())
        if L > best_len:
            best_i, best_len = i, L
    return best_i'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 25, "end_line": 28, "content": _CONTENT},
]
