"""Strong baseline: generous (near-full) input budget -- every source read in full.
Reference: vendor/text-simplification/baselines/truncation_full.py
"""

_FILE = "text-simplification/solution/truncation.py"

_CONTENT = '''    return 160'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 44, "end_line": 44, "content": _CONTENT},
]
