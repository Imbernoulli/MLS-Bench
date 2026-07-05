"""Mid baseline: nucleus (top-p) sampling, restricted but still no search.
Reference: vendor/text-simplification/baselines/strategy_topp.py
"""

_FILE = "text-simplification/solution/strategy.py"

_CONTENT = '''    return "topp"'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 41, "end_line": 41, "content": _CONTENT},
]
