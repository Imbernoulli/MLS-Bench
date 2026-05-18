"""Mid-edit operations for the opt-gradient-compression task.

Applied to the pytorch-vision workspace after pre_edit, before the agent starts.
Creates custom_compressor.py — the agent's editable gradient compression benchmark.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# ── Mid-edit operations ──────────────────────────────────────────────

OPS = [
    {
        "op": "create",
        "file": "pytorch-vision/custom_compressor.py",
        "content": _CUSTOM_PY,
    },
]
