"""Strong baseline: tight nucleus (restricted to the model's most probable tokens).
Reference: vendor/text-simplification/baselines/nucleus_tight.py
"""

_FILE = "text-simplification/solution/nucleus.py"

_CONTENT = '''    return 0.6'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 40, "end_line": 40, "content": _CONTENT},
]
