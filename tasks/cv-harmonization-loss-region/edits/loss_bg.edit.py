"""cv-harmonization-loss-region baseline: bg (background-only supervision — degenerate (weak)).
Reference: vendor/image-harmonization/baselines/loss_bg.py
"""

_FILE = "image-harmonization/solution/loss.py"

_CONTENT = "def get_loss_config():\n    return {'mode': 'bg'}"

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 19, "end_line": 21, "content": _CONTENT},
]
