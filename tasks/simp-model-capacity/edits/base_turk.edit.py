"""Strong baseline: t5-base checkpoint (more capacity), turk-focused fine-tune.
Reference: vendor/text-simplification/baselines/capacity_base_turk.py
"""

_FILE = "text-simplification/solution/capacity.py"

_CONTENT = '''    return "base_turk"'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 46, "end_line": 46, "content": _CONTENT},
]
