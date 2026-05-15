"""Mid-edit: create the editable template for meta-inner-loop-optimizer.

Applied to the learn2learn workspace after pre_edit, before the agent starts.
Creates custom_maml.py — the agent's editable algorithm file.
"""

from pathlib import Path

_TEMPLATE = Path(__file__).parent / "custom_template.py"
_CONTENT = _TEMPLATE.read_text()

OPS = [
    {
        "op": "create",
        "file": "learn2learn/custom_maml.py",
        "content": _CONTENT,
    },
]
