"""Strong baseline: real tuned-beam model decode.
Reference: vendor/machine-translation/baselines/strategy_beam.py
"""

_FILE = "machine-translation/solution/strategy.py"

_CONTENT = '''    return "beam"'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 11, "end_line": 11, "content": _CONTENT},
]
