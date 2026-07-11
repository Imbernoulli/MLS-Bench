"""Middle baseline: plain mean-log-prob normalization (length_penalty 1.0).
Reference: vendor/machine-translation/baselines/length_short.py (middle: plain norm)
"""

_FILE = "machine-translation/solution/length.py"

_CONTENT = '''    return {"length_penalty": 1.0, "min_length": 0, "max_new_tokens": 128}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 11, "end_line": 11, "content": _CONTENT},
]
