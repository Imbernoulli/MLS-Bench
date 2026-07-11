"""Unmeasured full-protocol candidate: prompt-length hardness weights."""

_FILE = "code-generation-lab/solution/policy_budget.py"
_CONTENT = '''def allocation_weights(problems):
    """Use prompt word count as a deterministic allocation signal."""
    return [float(max(1, len(problem["prompt"].split()))) for problem in problems]'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 5, "end_line": 7, "content": _CONTENT},
]
