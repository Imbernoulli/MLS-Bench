"""Strong baseline: no min-length floor (natural EOS stop point).
Reference: vendor/text-simplification/baselines/minlen_0.py
"""

_FILE = "text-simplification/solution/minlen.py"

_CONTENT = '''    return 0'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 40, "end_line": 40, "content": _CONTENT},
]
