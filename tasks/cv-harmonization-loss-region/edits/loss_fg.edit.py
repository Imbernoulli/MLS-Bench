"""cv-harmonization-loss-region baseline: fg (whole-image L1 + foreground emphasis — strong).
Reference: vendor/image-harmonization/baselines/loss_fg.py
"""

_FILE = "image-harmonization/solution/loss.py"

_CONTENT = "def get_loss_config():\n    return {'mode': 'fg'}"

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 19, "end_line": 21, "content": _CONTENT},
]
