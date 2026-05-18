"""Mid-edit operations for stf-traffic-forecast.
Creates custom_model.py and run.py in the BasicTS workspace root.
"""

from pathlib import Path

_EDITS_DIR = Path(__file__).parent
_CUSTOM_PY = (_EDITS_DIR / "custom_template.py").read_text()
_RUN_PY = (_EDITS_DIR / "run_template.py").read_text()

OPS = [
    {
        "op": "create",
        "file": "BasicTS/custom_model.py",
        "content": _CUSTOM_PY,
    },
    {
        "op": "create",
        "file": "BasicTS/run.py",
        "content": _RUN_PY,
    },
]
