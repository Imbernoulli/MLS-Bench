"""Mid-edit operations for the sr-symbolic-regression task.

Applied to the gplearn workspace after pre_edit, before the agent starts.
Creates custom_sr.py — the agent's editable algorithm file.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# ── Mid-edit operations ──────────────────────────────────────────────

OPS = [
    {
        "op": "create",
        "file": "gplearn/custom_sr.py",
        "content": _CUSTOM_PY,
    },
]
