"""Weak baseline: hot sampling temperature (near-random tokens).
Reference: vendor/text-simplification/baselines/temperature_hot.py
"""

_FILE = "text-simplification/solution/temperature.py"

_CONTENT = '''    return 2.0'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 41, "end_line": 41, "content": _CONTENT},
]
