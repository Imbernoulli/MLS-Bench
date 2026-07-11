"""Unmeasured full-protocol anchor candidate.

Retained for worker-side calibration. No score, ordering, or full-protocol runtime
is asserted before a complete 256x256, 100,000-step run.
"""

_FILE = "image-inpainting/solution/gate.py"

_CONTENT = '''    # `gate` is the precomputed learned sigmoid gate in [0,1]
    return feat * gate'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 17, "end_line": 17, "content": _CONTENT},
]
