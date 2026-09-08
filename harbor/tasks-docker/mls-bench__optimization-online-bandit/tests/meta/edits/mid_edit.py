"""Mid-edit operations for the opt-online-bandit task.

Applied to the SMPyBandits workspace after pre_edit, before the agent starts.
Creates custom_bandit.py — the agent's editable algorithm file.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# -- Mid-edit operations --------------------------------------------------

OPS = [
    {
        "op": "create",
        "file": "SMPyBandits/custom_bandit.py",
        "content": _CUSTOM_PY,
    },
]
