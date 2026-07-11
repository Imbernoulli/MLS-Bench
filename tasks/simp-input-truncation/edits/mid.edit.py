"""Mid baseline: moderate input budget.
Reference: vendor/text-simplification/baselines/truncation_mid.py
"""

_FILE = "text-simplification/solution/truncation.py"

_CONTENT = '''    return 48'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 44, "end_line": 44, "content": _CONTENT},
]
