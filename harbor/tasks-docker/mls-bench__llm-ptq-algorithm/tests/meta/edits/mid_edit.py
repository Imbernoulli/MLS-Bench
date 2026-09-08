"""Mid-edit: creates custom_ptq.py from template.

Applied to the gptq workspace after pre_edit, before the agent starts.
Creates the main quantization + evaluation script with editable regions.
"""
from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "gptq/custom_ptq.py",
        "content": _CUSTOM_PY,
    },
]
