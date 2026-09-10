"""Mid-edit operations for the opt-pac-bayes-bound task.

Applied to the PBB workspace after pre_edit, before the agent starts.
Creates custom_pac_bayes.py -- the agent's editable bound optimization file.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# -- Mid-edit operations ------------------------------------------------

OPS = [
    {
        "op": "create",
        "file": "PBB/custom_pac_bayes.py",
        "content": _CUSTOM_PY,
    },
]
