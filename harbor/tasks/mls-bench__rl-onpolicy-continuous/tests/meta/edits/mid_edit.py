"""Mid-edit operations for the rl-onpolicy-continuous task.

Applied to the cleanrl workspace after pre_edit, before the agent starts.
Creates custom_onpolicy_continuous.py — the agent's editable algorithm file.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# ── Mid-edit operations ──────────────────────────────────────────────

OPS = [
    {
        "op": "create",
        "file": "cleanrl/cleanrl/custom_onpolicy_continuous.py",
        "content": _CUSTOM_PY,
    },
]
