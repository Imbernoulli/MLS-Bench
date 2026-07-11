"""Strong baseline: tuned GNMT-style length penalty (~0.6 optimum).
Reference: vendor/machine-translation/baselines/length_tuned.py
"""

_FILE = "machine-translation/solution/length.py"

_CONTENT = '''    return {"length_penalty": 0.6, "min_length": 0, "max_new_tokens": 128}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 11, "end_line": 11, "content": _CONTENT},
]
