"""Mid-edit operations for llm-kv-adaptive-quantization."""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "transformers-kv-lab/custom_quant_eval.py",
        "content": _CUSTOM_PY,
    },
]
