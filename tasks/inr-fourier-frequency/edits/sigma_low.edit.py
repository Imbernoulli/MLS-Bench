"""Baseline edit for inr-fourier-frequency/sigma_low."""

_FILE = "inr-signal-fitting/solution/frequency.py"
_CONTENT = '''def surface_config():\n    return {"sigma": 0.3}'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 37, "end_line": 38, "content": _CONTENT},
]
