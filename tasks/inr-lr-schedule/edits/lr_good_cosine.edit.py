"""Baseline edit for inr-lr-schedule/lr_good_cosine."""

_FILE = "inr-signal-fitting/solution/lr_schedule.py"
_CONTENT = '''def surface_config():\n    return {"lr": 0.0005, "schedule": "cosine"}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 33, "end_line": 34, "content": _CONTENT},
]
