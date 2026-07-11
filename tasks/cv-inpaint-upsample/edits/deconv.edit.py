"""Unmeasured full-protocol anchor candidate.

Retained for worker-side calibration. No score, ordering, or full-protocol runtime
is asserted before a complete 256x256, 100,000-step run.
"""

_FILE = "image-inpainting/solution/upsample.py"

_CONTENT = '''    return nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1)'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 16, "end_line": 16, "content": _CONTENT},
]
