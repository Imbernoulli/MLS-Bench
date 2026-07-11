"""Unmeasured candidate: take the first candidate.

No selection: submit candidate 0. pass@1 equals the pool's first-sample pass@1.
This is best-of-N without a pool-dependent selection signal.
Reference surface: vendor/code-generation-lab/solution/policy_select.py
"""

_FILE = "code-generation-lab/solution/policy_select.py"

_CONTENT = '''def select_candidate(candidates, problem, tok):
    return 0'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 25, "end_line": 28, "content": _CONTENT},
]
