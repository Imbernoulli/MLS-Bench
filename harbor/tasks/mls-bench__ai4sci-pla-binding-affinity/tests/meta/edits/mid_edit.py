"""Mid-edit operations for pla-binding-affinity.
Creates EHIGN_PLA/custom_pla.py from template.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "EHIGN_PLA/custom_pla.py",
        "content": _CUSTOM_PY,
    },
]
