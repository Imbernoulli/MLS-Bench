"""Mid-edit operations for the rl-offline-off2on task.

Applied to the CORL workspace after pre_edit, before the agent starts.
Creates custom_finetune.py — the agent's editable algorithm file — from custom_template.py.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# ── Mid-edit operations ──────────────────────────────────────────────

OPS = [
    {
        "op": "create",
        "file": "CORL/algorithms/finetune/custom_finetune.py",
        "content": _CUSTOM_PY,
    },
]
