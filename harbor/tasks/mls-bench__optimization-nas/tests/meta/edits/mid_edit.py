"""Mid-edit operations for the automl-nas-search task.

Applied to the naslib workspace after pre_edit, before the agent starts.
Creates custom_nas_search.py — the agent's editable algorithm file.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# ── Mid-edit operations ──────────────────────────────────────────────

OPS = [
    {
        "op": "create",
        "file": "naslib/custom_nas_search.py",
        "content": _CUSTOM_PY,
    },
]
