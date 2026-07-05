"""Weak recurrence baseline for deblur-recurrence (the naive answer).

The naive / degenerate choice: a single full-res pass -> under-deblurs heavy blur. Reference: vendor/image-deblur/baselines/recur_one.py
"""

_FILE = "image-deblur/solution/recurrence.py"

_CONTENT = '''def get_recurrence_config():
    # a single full-res pass -> under-deblurs heavy blur
    return {"n_recurrence": 1}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
