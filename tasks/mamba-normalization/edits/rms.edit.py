"""Baseline edit for token-local RMS normalization."""

_FILE = "mamba/solution/normalization.py"
_CONTENT = '    return {"normalization": "rms"}'

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 7,
        "end_line": 7,
        "content": _CONTENT,
    },
]
