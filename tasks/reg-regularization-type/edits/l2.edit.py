"""Baseline for reg-regularization-type: STRONG: L2 diffusion (first-order gradient) regulariser — VoxelMorph default, smooth valid field.
Reference: vendor/deformable-registration/baselines/*.py
"""

_FILE = "deformable-registration/solution/regularization.py"

_CONTENT = 'def build_reg_type():\n    # STRONG: L2 diffusion (first-order gradient) regulariser — VoxelMorph default, smooth valid field.\n    return "l2"'

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 24,
        "end_line": 26,
        "content": _CONTENT,
    },
]
