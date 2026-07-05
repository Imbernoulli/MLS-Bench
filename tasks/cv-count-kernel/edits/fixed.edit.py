"""Weak baseline for cv-count-kernel: OVERSIZED fixed Gaussian kernel. Smears dense scenes -> higher MAE. Ref: vendor/crowd-counting/baselines/sigma_fixed.py"""

_FILE = "crowd-counting/solution/sigma.py"

_CONTENT = '    # Oversized fixed kernel (px): smears dense scenes.\n    return np.full((len(points),), 14.0, dtype=np.float32)'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 39, "end_line": 40, "content": _CONTENT},
]
