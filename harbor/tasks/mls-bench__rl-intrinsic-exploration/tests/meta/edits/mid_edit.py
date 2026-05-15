"""Mid-edit operations for the rl-intrinsic-exploration task."""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "cleanrl/cleanrl/custom_intrinsic_exploration.py",
        "content": _CUSTOM_PY,
    },
]
