"""Mid-edit: creates custom_loss.py from template."""
from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "pytorch-vision/custom_loss.py",
        "content": _CUSTOM_PY,
    },
]
