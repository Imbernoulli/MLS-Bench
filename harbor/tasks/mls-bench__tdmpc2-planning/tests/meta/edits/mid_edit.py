"""Mid-edit operations for the tdmpc2-planning task.

Applied after pre_edit, before the agent starts:
1. Creates custom_planner.py -- the agent's editable planning module
2. Modifies tdmpc2.py to import and delegate to custom_plan()

Operations are ordered bottom-to-top to avoid line-number shifts.
"""

from pathlib import Path

_DIR = Path(__file__).parent
_CUSTOM_PLANNER = (_DIR / "custom_template.py").read_text()

# Import and delegation code for tdmpc2.py
_IMPORT_LINE = "from common.custom_planner import custom_plan\n"

# Replacement for the _plan method body (lines 140-207 of tdmpc2.py)
_PLAN_DELEGATION = "\t@torch.no_grad()\n\tdef _plan(self, obs, t0=False, eval_mode=False, task=None):\n\t\t\"\"\"Delegate planning to custom_plan in custom_planner.py.\"\"\"\n\t\treturn custom_plan(self, obs, t0=t0, eval_mode=eval_mode, task=task)\n"

OPS = [
    # Create custom planner file
    {
        "op": "create",
        "file": "tdmpc2/tdmpc2/common/custom_planner.py",
        "content": _CUSTOM_PLANNER,
    },
    # Replace _plan method (lines 139-207) with delegation
    # Line 139 is @torch.no_grad() decorator, 140 is def _plan(...)
    {
        "op": "replace",
        "file": "tdmpc2/tdmpc2/tdmpc2.py",
        "start_line": 139,
        "end_line": 207,
        "content": _PLAN_DELEGATION,
    },
    # Insert import at top of tdmpc2.py (after line 7: from common.layers import ...)
    {
        "op": "insert",
        "file": "tdmpc2/tdmpc2/tdmpc2.py",
        "after_line": 7,
        "content": _IMPORT_LINE,
    },
]
