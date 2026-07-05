"""Mid baseline: t5-small checkpoint, turk-focused fine-tune.
Reference: vendor/text-simplification/baselines/capacity_small_turk.py
"""

_FILE = "text-simplification/solution/capacity.py"

_CONTENT = '''    return "small_turk"'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 47, "end_line": 47, "content": _CONTENT},
]
