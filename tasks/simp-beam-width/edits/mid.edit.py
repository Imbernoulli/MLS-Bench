"""Mid baseline: moderate beam width.
Reference: vendor/text-simplification/baselines/beamwidth_4.py
"""

_FILE = "text-simplification/solution/beamwidth.py"

_CONTENT = '''    return 4'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 38, "end_line": 38, "content": _CONTENT},
]
