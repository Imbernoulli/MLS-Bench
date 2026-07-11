"""Baseline edit for inr-hash-grid/pyramid16."""

_FILE = "inr-signal-fitting/solution/hash_grid.py"
_CONTENT = '''def surface_config():\n    return {"n_levels": 16, "base_res": 16, "finest_res": 256}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 36, "end_line": 37, "content": _CONTENT},
]
