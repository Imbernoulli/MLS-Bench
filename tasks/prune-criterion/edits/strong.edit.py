"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "prune-lab/solution/criterion.py"

_CONTENT = 'def importance(name, weight, grad):\n    # |weight| MAGNITUDE importance (Han et al., Deep Compression, 2015).\n    return weight.abs()\n'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 15, "end_line": 17, "content": _CONTENT},
]
