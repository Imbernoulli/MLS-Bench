"""Weak baseline: narrow beam width (under-searches).
Reference: vendor/text-simplification/baselines/beamwidth_2.py
"""

_FILE = "text-simplification/solution/beamwidth.py"

_CONTENT = '''    return 2'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 36, "end_line": 36, "content": _CONTENT},
]
