"""Mid-edit: create the evaluation harness for llm-dllm-demask-strategy."""

from pathlib import Path

_TEMPLATE = Path(__file__).parent / "custom_template.py"
_CONTENT = _TEMPLATE.read_text()

OPS = [
    {
        "op": "create",
        "file": "LLaDA/custom_demask_eval.py",
        "content": _CONTENT,
    },
]
