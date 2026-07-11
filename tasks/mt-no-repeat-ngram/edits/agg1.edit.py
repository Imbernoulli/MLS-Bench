"""Baseline agg1 for mt-no-repeat-ngram.
Reference: vendor/machine-translation/solution/norep.py
"""

_FILE = "machine-translation/solution/norep.py"

_CONTENT = '''    return {"no_repeat_ngram_size": 1}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 11, "end_line": 11, "content": _CONTENT},
]
