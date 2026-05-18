"""Mid-edit: create the optimization-convex-concave scaffold inside the package workspace."""

from pathlib import Path

_DIR = Path(__file__).parent
_CUSTOM_PY = (_DIR / "custom_template.py").read_text()
_FIXED_PY = (_DIR / "fixed_benchmark.py").read_text()

OPS = [
    {
        "op": "create",
        "file": "RAIN/optimization_convex_concave/custom_strategy.py",
        "content": _CUSTOM_PY,
    },
    {
        "op": "create",
        "file": "RAIN/optimization_convex_concave/fixed_benchmark.py",
        "content": _FIXED_PY,
    },
]
