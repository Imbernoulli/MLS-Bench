"""Unmeasured full-protocol candidate: equal hardness weights."""

_FILE = "code-generation-lab/solution/policy_budget.py"
_CONTENT = '''def allocation_weights(problems):
    """Return one non-negative hardness weight for every problem."""
    return [1.0 for _problem in problems]'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 5, "end_line": 7, "content": _CONTENT},
]
