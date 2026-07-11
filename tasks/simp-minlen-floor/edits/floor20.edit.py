"""Mid baseline: moderate min-length floor.
Reference: vendor/text-simplification/baselines/minlen_20.py
"""

_FILE = "text-simplification/solution/minlen.py"

_CONTENT = '''    return 20'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 42, "end_line": 42, "content": _CONTENT},
]
