"""Mid-edit operations for the irl-reward-learning task.

Applied to the imitation workspace after pre_edit, before the agent starts.
Creates custom_irl.py — the agent's editable algorithm file.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# ── Mid-edit operations ──────────────────────────────────────────────

OPS = [
    {
        "op": "create",
        "file": "imitation/custom_irl.py",
        "content": _CUSTOM_PY,
    },
]
