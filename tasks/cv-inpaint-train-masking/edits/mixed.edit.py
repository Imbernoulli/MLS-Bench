"""Unmeasured full-protocol anchor candidate.

Retained for worker-side calibration. No score, ordering, or full-protocol runtime
is asserted before a complete 256x256, 100,000-step run.
"""

_FILE = "image-inpainting/solution/masking.py"

_CONTENT = '''    import numpy as np
    def _sq(rng, size, lo, hi):
        for _ in range(300):
            frac = rng.uniform(lo, hi)
            s = int(round((frac ** 0.5) * size)); s = min(max(s, 4), size - 2)
            ty = int(rng.integers(0, size - s)); tx = int(rng.integers(0, size - s))
            m = np.zeros((size, size), np.float32); m[ty:ty + s, tx:tx + s] = 1.0
            if lo <= m.mean() <= hi:
                return m
        raise RuntimeError("cannot construct square mask")
    def _stroke(rng, size, lo, hi):
        for _ in range(80):
            m = np.zeros((size, size), np.float32)
            for _s in range(int(rng.integers(1, 4))):
                x = int(rng.integers(0, size)); y = int(rng.integers(0, size))
                ang = rng.uniform(0, 2 * np.pi)
                for _v in range(int(rng.integers(3, 7))):
                    ang += rng.uniform(-1.2, 1.2)
                    length = int(rng.integers(size // 12, size // 3)); w = int(rng.integers(size // 32, size // 12))
                    for t in range(length):
                        nx = int(round(x + np.cos(ang) * t)); ny = int(round(y + np.sin(ang) * t))
                        m[max(0, ny - w):min(size, ny + w + 1), max(0, nx - w):min(size, nx + w + 1)] = 1.0
                    x = int(round(x + np.cos(ang) * length)); y = int(round(y + np.sin(ang) * length))
                    x = min(max(x, 0), size - 1); y = min(max(y, 0), size - 1)
            if lo <= m.mean() <= hi:
                return m
        return _sq(rng, size, lo, hi)
    b, _, h, w = gt.shape
    holes = []
    for _ in range(b):
        if rng.uniform() < 0.70:
            holes.append(_sq(rng, h, 0.10, 0.25))
        else:
            holes.append(_stroke(rng, h, 0.12, 0.28))
    return torch.from_numpy(np.stack(holes)).unsqueeze(1).to(gt.device)'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 10, "end_line": 17, "content": _CONTENT},
]
