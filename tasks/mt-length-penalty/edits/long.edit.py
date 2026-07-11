"""Baseline with length_penalty=2.0.
Reference: vendor/machine-translation/baselines/length_short.py
"""

_FILE = "machine-translation/solution/length.py"

_CONTENT = '''    return {"length_penalty": 2.0}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 11, "end_line": 11, "content": _CONTENT},
]
