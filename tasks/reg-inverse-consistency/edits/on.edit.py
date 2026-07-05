"""Baseline for reg-inverse-consistency: MID: inverse/cycle-consistency (inverse_w=1.0, SYMNet-style) — symmetric & fold-free but trades away some alignment.
Reference: vendor/deformable-registration/baselines/*.py
"""

_FILE = "deformable-registration/solution/inverse.py"

_CONTENT = 'def build_inverse_weight():\n    # MID: inverse/cycle-consistency (inverse_w=1.0, SYMNet-style) — symmetric & fold-free but trades away some alignment.\n    return 1.0'

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 23,
        "end_line": 25,
        "content": _CONTENT,
    },
]
