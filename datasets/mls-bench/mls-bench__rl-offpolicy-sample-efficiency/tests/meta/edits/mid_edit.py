"""Mid-edit operations for rl-offpolicy-sample-efficiency task.

Creates custom_algorithm.py in the FastTD3 workspace from custom_template.py.
This file is the agent's editable training script.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "FastTD3/fast_td3/custom_algorithm.py",
        "content": _CUSTOM_PY,
    },
]
