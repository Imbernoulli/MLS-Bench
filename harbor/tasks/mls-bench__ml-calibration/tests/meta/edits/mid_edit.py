"""Mid-edit: creates custom_calibration.py from template."""
from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "scikit-learn/custom_calibration.py",
        "content": _CUSTOM_PY,
    },
]
