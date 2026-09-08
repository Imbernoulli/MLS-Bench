"""Mid-edit operations for the tdmpc2-simnorm task.

Applied after pre_edit, before the agent starts:
1. Creates custom_simnorm.py — the agent's editable normalization module
2. Replaces SimNorm in layers.py with import from custom_simnorm.py

Operations are ordered bottom-to-top to avoid line-number shifts.
"""

from pathlib import Path

_DIR = Path(__file__).parent
_CUSTOM_SIMNORM = (_DIR / "custom_template.py").read_text()

# Import statement that replaces the SimNorm class in layers.py
_IMPORT_CUSTOM = """\
from common.custom_simnorm import CustomSimNorm as SimNorm
"""

OPS = [
    # Create custom normalization file
    {
        "op": "create",
        "file": "tdmpc2/tdmpc2/common/custom_simnorm.py",
        "content": _CUSTOM_SIMNORM,
    },
    # Replace SimNorm class (lines 74-91) with import from custom file
    {
        "op": "replace",
        "file": "tdmpc2/tdmpc2/common/layers.py",
        "start_line": 74,
        "end_line": 91,
        "content": _IMPORT_CUSTOM,
    },
]
