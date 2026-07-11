"""Truncation baseline: keep the first 75% of the words (naive tail deletion).
Reference: vendor/text-simplification/baselines/policy_truncate.py
"""

_FILE = "text-simplification/solution/policy.py"

_CONTENT = '''    return "truncate"'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 34, "end_line": 35, "content": _CONTENT},
]
