"""Mid-edit operations for ai4bio-protein-inverse-folding.
Creates ProteinInvBench/custom_invfold.py from template.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "ProteinInvBench/custom_invfold.py",
        "content": _CUSTOM_PY,
    },
]
