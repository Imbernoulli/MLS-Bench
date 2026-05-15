"""Mid-edit operations for mol-property-prediction.
Creates Uni-Mol/custom_molprop.py from template.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "Uni-Mol/custom_molprop.py",
        "content": _CUSTOM_PY,
    },
]
