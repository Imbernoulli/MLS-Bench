"""Weak baseline: plain multinomial sampling, no search at all.
Reference: vendor/text-simplification/baselines/strategy_sample.py
"""

_FILE = "text-simplification/solution/strategy.py"

_CONTENT = '''    return "sample"'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 41, "end_line": 41, "content": _CONTENT},
]
