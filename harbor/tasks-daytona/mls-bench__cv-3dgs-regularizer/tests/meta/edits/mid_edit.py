"""Mid-edit: create custom_regularizer.py and train_gsplat.py in gsplat workspace."""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_TRAIN_PATH = Path(__file__).parent / "train_gsplat.py"

OPS = [
    {
        "op": "create",
        "file": "gsplat/custom_regularizer.py",
        "content": _TEMPLATE_PATH.read_text(),
    },
    {
        "op": "create",
        "file": "gsplat/train_gsplat.py",
        "content": _TRAIN_PATH.read_text(),
    },
]
