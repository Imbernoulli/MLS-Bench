"""Degenerate floor baseline: emit empty string (no simplification -> low SARI).
Reference: vendor/text-simplification/baselines/policy_empty.py
"""

_FILE = "text-simplification/solution/policy.py"

_CONTENT = '''    return "empty"'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 39, "end_line": 39, "content": _CONTENT},
]
