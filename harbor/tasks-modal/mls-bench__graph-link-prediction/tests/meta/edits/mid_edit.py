"""Mid-edit operations for graph-link-prediction.
Creates pytorch-geometric-lp/custom_linkpred.py from template.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "pytorch-geometric-lp/custom_linkpred.py",
        "content": _CUSTOM_PY,
    },
]
