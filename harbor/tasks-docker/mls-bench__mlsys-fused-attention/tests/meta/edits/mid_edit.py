"""Mid-edit: create the benchmark script for mlsys-fused-attention."""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "flash-attention/custom_triton_bench.py",
        "content": _CUSTOM_PY,
    },
]
