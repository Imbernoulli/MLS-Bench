"""Mid-edit operations for the ml-selective-deferral task."""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "scikit-learn/custom_selective.py",
        "content": _CUSTOM_PY,
    },
]
