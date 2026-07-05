"""mono3d-uncertainty-weighting baseline: weight_homoscedastic.

Auto-generated from vendor/mono3d-detection/baselines/weight_homoscedastic.py. Replaces the editable region of
mono3d-detection/solution/task_weighting.py (the `build_task_weighting` surface) with the weight_homoscedastic implementation.
"""

_FILE = "mono3d-detection/solution/task_weighting.py"

_CONTENT = 'def build_task_weighting():\n    class _W(nn.Module):\n        def __init__(self):\n            super().__init__()\n            self.log_sigma = nn.Parameter(torch.zeros(3))\n\n    mod = _W()\n\n    def weight(losses):\n        s = mod.log_sigma\n        keys = ["depth", "orient", "dims"]\n        total = 0.0\n        for i, k in enumerate(keys):\n            total = total + torch.exp(-s[i]) * losses[k] + s[i]\n        return total\n\n    return mod, weight'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 29, "end_line": 35, "content": _CONTENT},
]
