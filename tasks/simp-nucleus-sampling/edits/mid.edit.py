"""Mid baseline: moderately restricted nucleus.
Reference: vendor/text-simplification/baselines/nucleus_mid.py
"""

_FILE = "text-simplification/solution/nucleus.py"

_CONTENT = '''    return 0.9'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 40, "end_line": 40, "content": _CONTENT},
]
