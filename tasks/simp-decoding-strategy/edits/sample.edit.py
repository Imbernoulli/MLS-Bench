"""Weak baseline: plain multinomial sampling, no search at all.
Reference: vendor/text-simplification/baselines/strategy_sample.py
"""

_FILE = "text-simplification/solution/strategy.py"

_CONTENT = '''    return "sample"'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 43, "end_line": 43, "content": _CONTENT},
]
