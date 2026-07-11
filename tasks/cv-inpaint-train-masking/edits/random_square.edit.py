"""Unmeasured full-protocol anchor candidate.

Retained for worker-side calibration. No score, ordering, or full-protocol runtime
is asserted before a complete 256x256, 100,000-step run.
"""

_FILE = "image-inpainting/solution/masking.py"

_CONTENT = '''    import numpy as np
    def _square_hole(rng, size, lo, hi):
        for _ in range(300):
            frac = rng.uniform(lo, hi)
            s = int(round((frac ** 0.5) * size)); s = min(max(s, 4), size - 2)
            ty = int(rng.integers(0, size - s)); tx = int(rng.integers(0, size - s))
            m = np.zeros((size, size), np.float32); m[ty:ty + s, tx:tx + s] = 1.0
            if lo <= m.mean() <= hi:
                return m
        return m
    b, _, h, w = gt.shape
    holes = np.stack([_square_hole(rng, h, 0.10, 0.25) for _ in range(b)])
    return torch.from_numpy(holes).unsqueeze(1).to(gt.device)'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 10, "end_line": 17, "content": _CONTENT},
]
