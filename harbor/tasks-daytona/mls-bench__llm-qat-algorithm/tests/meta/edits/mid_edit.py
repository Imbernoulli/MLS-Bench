"""Mid-edit: creates custom_qat.py from template for the llm-qat-algorithm task.

Applied to the llm-qat-runtime workspace after pre_edit, before the agent
or any baseline starts.  Produces the QAT finetune + evaluation script
with editable regions clearly marked.
"""
from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "llm-qat-runtime/custom_qat.py",
        "content": _CUSTOM_PY,
    },
]
