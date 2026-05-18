"""Mid-edit for cv-vae-loss.

Creates custom_train.py from the template.
"""

from pathlib import Path

_TEMPLATE = (Path(__file__).parent / "custom_template.py").read_text()

OPS = [
    {
        "op": "create",
        "file": "diffusers-main/custom_train.py",
        "content": _TEMPLATE,
    },
]
