"""Mid-edit: create the editable template for mlsys-moe-load-balance."""

from pathlib import Path

_TEMPLATE = Path(__file__).parent / "custom_template.py"
_CONTENT = _TEMPLATE.read_text()

OPS = [
    {
        "op": "create",
        "file": "eplb/custom_eplb.py",
        "content": _CONTENT,
    },
]
