"""Mid-edit operations for the jepa-prediction-loss task.

Applied to the eb_jepa workspace after pre_edit, before the agent starts.
Creates custom_prediction_loss.py -- the agent's editable training file --
from custom_template.py.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# -- Mid-edit operations ---------------------------------------------------

OPS = [
    {
        "op": "create",
        "file": "eb_jepa/custom_prediction_loss.py",
        "content": _CUSTOM_PY,
    },
]
