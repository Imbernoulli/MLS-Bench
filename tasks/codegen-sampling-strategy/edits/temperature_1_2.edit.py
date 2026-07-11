"""Unmeasured full-protocol candidate: temperature 1.2, top-p 0.95."""

_FILE = "code-generation-lab/solution/policy_sampling.py"
_CONTENT = '''def sampling_parameters(problem):
    """Return ``(temperature, top_p)`` for one policy-visible problem."""
    return 1.2, 0.95'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 10, "end_line": 12, "content": _CONTENT},
]
