"""Baseline lower for mt-postprocess-detok.
Reference: vendor/machine-translation/solution/postproc.py
"""

_FILE = "machine-translation/solution/postproc.py"

_CONTENT = '''    return "lowercase"'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 11, "end_line": 11, "content": _CONTENT},
]
