"""Mid-edit operations for ai4bio-mutation-effect-prediction.
Creates ProteinGym/custom_mutation_pred.py from template.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "ProteinGym/custom_mutation_pred.py",
        "content": _CUSTOM_PY,
    },
]
