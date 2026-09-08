"""Mid-edit: create the editable template for ml-dimensionality-reduction.

Creates the benchmark scaffold inside the scikit-learn workspace:
  scikit-learn/bench/custom_dimred.py  -- agent-editable reducer + evaluation harness
"""

from pathlib import Path

_TEMPLATE = Path(__file__).parent / "custom_template.py"
_CONTENT = _TEMPLATE.read_text()

OPS = [
    {
        "op": "create",
        "file": "scikit-learn/bench/custom_dimred.py",
        "content": _CONTENT,
    },
]
