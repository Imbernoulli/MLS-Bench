"""Mid-edit operations for dlm-dkv-policy."""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "dLLM-cache/custom_dlm_eval.py",
        "content": _CUSTOM_PY,
    },
]
