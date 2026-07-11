"""Weak baseline: wide nucleus (unrestricted, near-full-distribution sampling).
Reference: vendor/text-simplification/baselines/nucleus_wide.py
"""

_FILE = "text-simplification/solution/nucleus.py"

_CONTENT = '''    return 1.0'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 40, "end_line": 40, "content": _CONTENT},
]
