"""Unmeasured full-protocol candidate: prompt-length token-cap weights."""

_FILE = "code-generation-lab/solution/policy_decode.py"
_CONTENT = '''def token_cap_weights(problems):
    """Use prompt word count as a deterministic length-need signal."""
    return [float(max(1, len(problem["prompt"].split()))) for problem in problems]'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 5, "end_line": 7, "content": _CONTENT},
]
