"""Baseline for reg-regularization-type: WEAK: no regulariser — the field folds (non-diffeomorphic), TRE/PSNR degrade at large warps.
Reference: vendor/deformable-registration/baselines/*.py
"""

_FILE = "deformable-registration/solution/regularization.py"

_CONTENT = 'def build_reg_type():\n    # WEAK: no regulariser — the field folds (non-diffeomorphic), TRE/PSNR degrade at large warps.\n    return "none"'

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 24,
        "end_line": 26,
        "content": _CONTENT,
    },
]
