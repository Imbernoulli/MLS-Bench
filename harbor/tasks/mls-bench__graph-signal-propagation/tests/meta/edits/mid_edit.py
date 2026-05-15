"""Mid-edit operations for the graph-signal-propagation task.

Applied to the ChebNetII workspace after pre_edit, before the agent starts.
Creates custom_filter.py -- the agent's editable algorithm file.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "ChebNetII/main/custom_filter.py",
        "content": _CUSTOM_PY,
    },
]
