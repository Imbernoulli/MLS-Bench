"""Mid-edit: create the editable template for opt-multi-objective."""

from pathlib import Path

_TEMPLATE = Path(__file__).parent / "custom_template.py"
_CONTENT = _TEMPLATE.read_text()

OPS = [
    {
        "op": "create",
        "file": "deap/custom_moea.py",
        "content": _CONTENT,
    },
]
