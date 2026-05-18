"""Mid-edit operations for the causal-treatment-effect task.

Applied to the scikit-learn workspace after pre_edit, before the agent starts.
Creates custom_cate.py -- the agent's editable algorithm file.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# -- Mid-edit operations --

OPS = [
    {
        "op": "create",
        "file": "scikit-learn/custom_cate.py",
        "content": _CUSTOM_PY,
    },
]
