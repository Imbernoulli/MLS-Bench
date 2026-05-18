"""Mid-edit operations for ml-clustering-algorithm.

Creates the custom clustering script in the scikit-learn workspace.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "scikit-learn/custom_clustering.py",
        "content": _CUSTOM_PY,
    },
]
