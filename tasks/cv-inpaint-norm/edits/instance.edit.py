"""Unmeasured full-protocol anchor candidate.

Retained for worker-side calibration. No score, ordering, or full-protocol runtime
is asserted before a complete 256x256, 100,000-step run.
"""

_FILE = "image-inpainting/solution/norm.py"

_CONTENT = '''    return nn.InstanceNorm2d(num_ch, affine=True)'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 17, "end_line": 17, "content": _CONTENT},
]
