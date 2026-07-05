"""Baseline for reg-inverse-consistency: WEAK: over-strong inverse-consistency (inverse_w=50) — over-constrains, collapses the deformation.
Reference: vendor/deformable-registration/baselines/*.py
"""

_FILE = "deformable-registration/solution/inverse.py"

_CONTENT = 'def build_inverse_weight():\n    # WEAK: over-strong inverse-consistency (inverse_w=50) — over-constrains, collapses the deformation.\n    return 50.0'

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 23,
        "end_line": 25,
        "content": _CONTENT,
    },
]
