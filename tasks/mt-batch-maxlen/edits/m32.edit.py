"""Baseline m32 for mt-batch-maxlen.
Reference: vendor/machine-translation/solution/maxlen.py
"""

_FILE = "machine-translation/solution/maxlen.py"

_CONTENT = '''    return 32'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 11, "end_line": 11, "content": _CONTENT},
]
