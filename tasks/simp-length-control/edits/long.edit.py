"""Weak baseline: runaway-long window -> under-compressed (near copy-input).
Reference: vendor/text-simplification/baselines/length_long.py
"""

_FILE = "text-simplification/solution/length.py"

_CONTENT = '''    return {"min_length": 40, "max_length": 160, "length_penalty": 2.5}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 40, "end_line": 40, "content": _CONTENT},
]
