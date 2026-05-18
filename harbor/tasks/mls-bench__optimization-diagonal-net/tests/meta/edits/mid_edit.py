"""Mid-edit: create the opt-diagonal-net scaffold inside the RAIN workspace."""

from pathlib import Path

_DIR = Path(__file__).parent
_CUSTOM_PY = (_DIR / "custom_template.py").read_text()
_FIXED_PY = (_DIR / "fixed_benchmark.py").read_text()

OPS = [
    {
        "op": "create",
        "file": "RAIN/opt_diagonal_net/custom_optimizer.py",
        "content": _CUSTOM_PY,
    },
    {
        "op": "create",
        "file": "RAIN/opt_diagonal_net/fixed_benchmark.py",
        "content": _FIXED_PY,
    },
]
