"""Mid-edit operations for the opt-dp-sgd task.

Applied to the opacus workspace after pre_edit, before the agent starts.
Creates custom_dpsgd.py — the agent's editable DP-SGD training file — from custom_template.py.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# -- Mid-edit operations ---------------------------------------------------

OPS = [
    {
        "op": "create",
        "file": "opacus/custom_dpsgd.py",
        "content": _CUSTOM_PY,
    },
]
