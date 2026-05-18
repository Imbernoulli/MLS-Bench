"""Mid-edit operations for ai4bio-protein-structure-repr.
Creates ProteinWorkshop/custom_protein_encoder.py from template.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "ProteinWorkshop/custom_protein_encoder.py",
        "content": _CUSTOM_PY,
    },
]
