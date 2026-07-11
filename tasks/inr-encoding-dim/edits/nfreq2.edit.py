"""Baseline edit for inr-encoding-dim/nfreq2."""

_FILE = "inr-signal-fitting/solution/encoding_dim.py"
_CONTENT = '''def surface_config():\n    return {"num_freqs": 2}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 34, "end_line": 35, "content": _CONTENT},
]
