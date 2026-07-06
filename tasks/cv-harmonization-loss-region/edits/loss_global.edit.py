"""cv-harmonization-loss-region baseline: global (whole-image L1).
Reference: vendor/image-harmonization/baselines/loss_global.py
"""

_FILE = "image-harmonization/solution/loss.py"

_CONTENT = "def get_loss_config():\n    return {'mode': 'global'}"

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 19, "end_line": 21, "content": _CONTENT},
]
