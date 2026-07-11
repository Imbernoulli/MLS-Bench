"""Baseline edit for inr-fourier-frequency/sigma_high."""

_FILE = "inr-signal-fitting/solution/frequency.py"
_CONTENT = '''def surface_config():\n    return {"sigma": 100.0}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 37, "end_line": 38, "content": _CONTENT},
]
