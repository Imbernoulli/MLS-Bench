"""Mid-edit: Create custom_strategy.py and train_gsplat.py in gsplat workspace."""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_TEMPLATE = _TEMPLATE_PATH.read_text()

_TRAIN_PATH = Path(__file__).parent / "train_gsplat.py"
_TRAIN_SCRIPT = _TRAIN_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "gsplat/custom_strategy.py",
        "content": _CUSTOM_TEMPLATE,
    },
    {
        "op": "create",
        "file": "gsplat/train_gsplat.py",
        "content": _TRAIN_SCRIPT,
    },
]
