"""Weak baseline: aggressively short input budget (silently drops tail content).
Reference: vendor/text-simplification/baselines/truncation_short.py
"""

_FILE = "text-simplification/solution/truncation.py"

_CONTENT = '''    return 16'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 44, "end_line": 44, "content": _CONTENT},
]
