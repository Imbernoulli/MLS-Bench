"""Unmeasured full-protocol anchor candidate.

Retained for worker-side calibration. No score, ordering, or full-protocol runtime
is asserted before a complete 256x256, 100,000-step run.
"""

_FILE = "image-inpainting/solution/loss.py"

_CONTENT = '''    valid = 1.0 - mask
    return (torch.abs(out - gt) * valid).sum() / (valid.sum() + 1e-8)'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 15, "end_line": 16, "content": _CONTENT},
]
