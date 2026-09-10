"""Mid-edit operations for the meta-fewshot-classification task.

Applied to the easy-few-shot-learning workspace after pre_edit, before the agent starts.
Creates custom_fewshot.py — the agent's editable algorithm file.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# ── Mid-edit operations ──────────────────────────────────────────────

OPS = [
    {
        "op": "create",
        "file": "easy-few-shot-learning/custom_fewshot.py",
        "content": _CUSTOM_PY,
    },
]
