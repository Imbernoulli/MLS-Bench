"""Strong baseline: wide beam width.
Reference: vendor/text-simplification/baselines/beamwidth_8.py
"""

_FILE = "text-simplification/solution/beamwidth.py"

_CONTENT = '''    return 8'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 36, "end_line": 36, "content": _CONTENT},
]
