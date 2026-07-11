"""Unmeasured full-protocol candidate: equal token-cap weights."""

_FILE = "code-generation-lab/solution/policy_decode.py"
_CONTENT = '''def token_cap_weights(problems):
    """Return one non-negative length-need weight for every problem."""
    return [1.0 for _problem in problems]'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 5, "end_line": 7, "content": _CONTENT},
]
