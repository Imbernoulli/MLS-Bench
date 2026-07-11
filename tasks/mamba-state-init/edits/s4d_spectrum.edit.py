"""Baseline edit for the S4D real decay spectrum."""

OPS = [{
    "op": "replace",
    "file": "mamba/solution/state_init.py",
    "start_line": 7,
    "end_line": 7,
    "content": '    return {"scheme": "s4d_spectrum"}',
}]
