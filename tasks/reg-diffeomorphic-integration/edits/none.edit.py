"""Baseline for reg-diffeomorphic-integration: WEAK: no integration (steps=0) — plain displacement, field can FOLD (non-diffeomorphic).
Reference: vendor/deformable-registration/baselines/*.py
"""

_FILE = "deformable-registration/solution/integration.py"

_CONTENT = 'def build_integration_steps():\n    # WEAK: no integration (steps=0) — plain displacement, field can FOLD (non-diffeomorphic).\n    return 0'

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 25,
        "end_line": 27,
        "content": _CONTENT,
    },
]
