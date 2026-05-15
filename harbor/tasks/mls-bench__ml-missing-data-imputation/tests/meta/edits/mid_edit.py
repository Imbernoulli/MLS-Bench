"""Mid-edit operations for ml-missing-data-imputation.

Creates the custom imputation script in the scikit-learn workspace.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "scikit-learn/custom_imputation.py",
        "content": _CUSTOM_PY,
    },
]
