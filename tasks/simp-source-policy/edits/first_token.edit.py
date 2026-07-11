"""Degenerate floor baseline: emit only the first source word (low SARI).
Reference: vendor/text-simplification/baselines/policy_first_token.py
"""

_FILE = "text-simplification/solution/policy.py"

_CONTENT = '''    return "first_token"'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 34, "end_line": 35, "content": _CONTENT},
]
