"""Baseline never for mt-early-stopping.
Reference: vendor/machine-translation/solution/earlystop.py
"""

_FILE = "machine-translation/solution/earlystop.py"

_CONTENT = '''    return "never"'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 11, "end_line": 11, "content": _CONTENT},
]
