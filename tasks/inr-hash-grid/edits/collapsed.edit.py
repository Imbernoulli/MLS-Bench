"""Baseline edit for inr-hash-grid/collapsed."""

_FILE = "inr-signal-fitting/solution/hash_grid.py"
_CONTENT = '''def surface_config():\n    return {"n_levels": 1, "base_res": 4, "finest_res": 4}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 36, "end_line": 37, "content": _CONTENT},
]
