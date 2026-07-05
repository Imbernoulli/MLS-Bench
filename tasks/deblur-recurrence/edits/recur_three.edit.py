"""Strong recurrence baseline for deblur-recurrence (the good answer).

The strong reference: 4 shared-weight refinement passes (SRN recurrence) -> sharper. Reference: vendor/image-deblur/baselines/recur_three.py
"""

_FILE = "image-deblur/solution/recurrence.py"

_CONTENT = '''def get_recurrence_config():
    # 4 shared-weight refinement passes (SRN recurrence) -> sharper
    return {"n_recurrence": 4}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 33, "content": _CONTENT},
]
