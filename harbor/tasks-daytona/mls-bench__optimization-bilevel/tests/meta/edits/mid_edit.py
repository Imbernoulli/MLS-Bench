"""Mid-edit: create the optimization-bilevel scaffold inside the package workspace."""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "penalized-bilevel-gradient-descent/mlsbench/custom_strategy.py",
        "content": _CUSTOM_PY,
    },
]
