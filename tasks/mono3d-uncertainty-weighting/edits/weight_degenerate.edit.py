"""mono3d-uncertainty-weighting baseline: weight_degenerate.

Auto-generated from vendor/mono3d-detection/baselines/weight_degenerate.py. Replaces the editable region of
mono3d-detection/solution/task_weighting.py (the `build_task_weighting` surface) with the weight_degenerate implementation.
"""

_FILE = "mono3d-detection/solution/task_weighting.py"

_CONTENT = 'def build_task_weighting():\n    def weight(losses):\n        # starve depth + dims (0.001), over-weight orient -> untrained H/residual, broken depth.\n        return 0.001 * losses["depth"] + losses["orient"] + 0.001 * losses["dims"]\n\n    return nn.Identity(), weight'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 29, "end_line": 35, "content": _CONTENT},
]
