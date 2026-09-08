"""Mid-edit operations for the safe-rl task.

Applied to the omnisafe workspace after pre_edit, before the agent starts.
Creates custom_lag.py -- the agent's editable algorithm file.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# -- Mid-edit operations --

OPS = [
    {
        "op": "create",
        "file": "omnisafe/omnisafe/algorithms/on_policy/naive_lagrange/custom_lag.py",
        "content": _CUSTOM_PY,
    },
]
