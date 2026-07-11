"""Baseline tuned for mt-repetition-penalty.
Reference: vendor/machine-translation/solution/reppen.py
"""

_FILE = "machine-translation/solution/reppen.py"

_CONTENT = '''    return {"repetition_penalty": 1.1}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 11, "end_line": 11, "content": _CONTENT},
]
