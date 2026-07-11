"""Strong baseline: sensibly compressive window (SARI-optimal).
Reference: vendor/text-simplification/baselines/length_tuned.py
"""

_FILE = "text-simplification/solution/length.py"

_CONTENT = '''    return {"min_length": 0, "max_length": 96, "length_penalty": 1.0}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 42, "end_line": 42, "content": _CONTENT},
]
