"""Degenerate baseline: copy the German source unchanged (~0 BLEU).
Reference: vendor/machine-translation/baselines/strategy_copy_source.py
"""

_FILE = "machine-translation/solution/strategy.py"

_CONTENT = '''    return "copy_source"'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 11, "end_line": 11, "content": _CONTENT},
]
