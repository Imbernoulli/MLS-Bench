"""Mid-edit operations for the fed-aggregation-strategy task.

Applied to the flower workspace after pre_edit, before the agent starts.
Creates custom_fl_aggregation.py — the agent's editable algorithm file.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# -- Mid-edit operations ------------------------------------------------

OPS = [
    {
        "op": "create",
        "file": "flower/custom_fl_aggregation.py",
        "content": _CUSTOM_PY,
    },
]
