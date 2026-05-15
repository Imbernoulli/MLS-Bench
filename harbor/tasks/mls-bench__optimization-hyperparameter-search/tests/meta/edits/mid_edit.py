"""Mid-edit: create the editable template for opt-hyperparameter-search."""

from pathlib import Path

_TEMPLATE = Path(__file__).parent / "custom_template.py"
_CONTENT = _TEMPLATE.read_text()

OPS = [
    {
        "op": "create",
        "file": "scikit-learn/custom_hpo.py",
        "content": _CONTENT,
    },
]
