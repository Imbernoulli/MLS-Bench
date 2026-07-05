"""Weak baseline: large min-length floor (forces padding past natural EOS).
Reference: vendor/text-simplification/baselines/minlen_60.py
"""

_FILE = "text-simplification/solution/minlen.py"

_CONTENT = '''    return 60'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 40, "end_line": 40, "content": _CONTENT},
]
