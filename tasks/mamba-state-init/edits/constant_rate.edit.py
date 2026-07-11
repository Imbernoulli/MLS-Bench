"""Baseline edit for a constant A=-1 decay rate."""

OPS = [{
    "op": "replace",
    "file": "mamba/solution/state_init.py",
    "start_line": 7,
    "end_line": 7,
    "content": '    return {"scheme": "constant_rate"}',
}]
