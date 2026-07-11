"""Strong baseline: deterministic beam search, real sequence-probability search.
Reference: vendor/text-simplification/baselines/strategy_beam.py
"""

_FILE = "text-simplification/solution/strategy.py"

_CONTENT = '''    return "beam"'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 43, "end_line": 43, "content": _CONTENT},
]
