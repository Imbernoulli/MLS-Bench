"""Weak baseline: PICK-FIRST — return candidates[0], spending no dev budget.

No search at all: commit to the first candidate. With a mixed candidate pool this
often lands on a mediocre (or misleading) instruction and generalizes poorly to the
held-out TEST set. Reference: vendor/prompt-optimization-lab/solution/strategy.py.
"""

_FILE = "prompt-optimization-lab/solution/strategy.py"

_CONTENT = '''def select(candidates, ctx) -> str:
    # No search: commit to the first candidate.
    return candidates[0]'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 45, "end_line": 47, "content": _CONTENT},
]
