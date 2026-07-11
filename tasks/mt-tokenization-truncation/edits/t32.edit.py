"""Baseline t32 for mt-tokenization-truncation.
Reference: vendor/machine-translation/solution/tok.py
"""

_FILE = "machine-translation/solution/tok.py"

_CONTENT = '''    return 32'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 11, "end_line": 11, "content": _CONTENT},
]
