"""Unmeasured full-protocol anchor candidate.

Retained for worker-side calibration. No score, ordering, or full-protocol runtime
is asserted before a complete 256x256, 100,000-step run.
"""

_FILE = "image-inpainting/solution/masking.py"

_CONTENT = '''    b, _, h, w = gt.shape
    side = int((0.05 ** 0.5) * min(h, w)) + 1
    masks = torch.zeros(b, 1, h, w, device=gt.device)
    masks[:, :, :side, :side] = 1.0
    return masks'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 10, "end_line": 17, "content": _CONTENT},
]
