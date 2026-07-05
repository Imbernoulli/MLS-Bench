"""Good baseline for cv-count-kernel: geometry-ADAPTIVE k-NN Gaussian kernel (MCNN/CSRNet, beta~0.3). Resolves dense scenes -> lower MAE. Ref: vendor/crowd-counting/baselines/sigma_adaptive.py"""

_FILE = "crowd-counting/solution/sigma.py"

_CONTENT = '    beta, k = 0.3, 3\n    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)\n    n = len(pts)\n    if n == 0:\n        return np.zeros((0,), dtype=np.float32)\n    if n == 1:\n        return np.array([6.0], dtype=np.float32)\n    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)\n    np.fill_diagonal(d2, np.inf)\n    kk = min(k, n - 1)\n    nn_d = np.sqrt(np.sort(d2, axis=1)[:, :kk]).mean(axis=1)\n    sig = beta * nn_d\n    return np.clip(sig, 2.0, 12.0).astype(np.float32)'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 39, "end_line": 40, "content": _CONTENT},
]
