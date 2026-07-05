"""Baseline for reg-regularization-type: STRONG-ALT: bending-energy (second-order TPS/B-spline) regulariser — penalises curvature only.
Reference: vendor/deformable-registration/baselines/*.py
"""

_FILE = "deformable-registration/solution/regularization.py"

_CONTENT = 'def build_reg_type():\n    # STRONG-ALT: bending-energy (second-order TPS/B-spline) regulariser — penalises curvature only.\n    return "bending"'

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 24,
        "end_line": 26,
        "content": _CONTENT,
    },
]
