"""Mid-edit operations for the jepa-planning task.

Applied to the eb_jepa workspace after pre_edit, before the agent starts.
Creates custom_planner.py -- the agent's editable planning script -- from custom_template.py.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# -- Mid-edit operations --

OPS = [
    {
        "op": "create",
        "file": "eb_jepa/custom_planner.py",
        "content": _CUSTOM_PY,
    },
]
