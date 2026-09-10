"""Mid-edit operations for opt-evolution-strategy.

Creates custom_evolution.py — the agent's editable optimization script —
from custom_template.py. Placed into the deap package directory.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "deap/custom_evolution.py",
        "content": _CUSTOM_PY,
    },
]
