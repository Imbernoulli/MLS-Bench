"""Mid-edit operations for the graph-node-classification task.

Applied to the pytorch-geometric workspace after pre_edit, before the agent starts.
Creates custom_nodecls.py — the agent's editable algorithm file.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# ── Mid-edit operations ──────────────────────────────────────────────

OPS = [
    {
        "op": "create",
        "file": "pytorch-geometric/custom_nodecls.py",
        "content": _CUSTOM_PY,
    },
]
