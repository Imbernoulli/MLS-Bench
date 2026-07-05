"""Baseline for reg-diffeomorphic-integration: MID: light integration (steps=3) — partly diffeomorphic, some folding removed.
Reference: vendor/deformable-registration/baselines/*.py
"""

_FILE = "deformable-registration/solution/integration.py"

_CONTENT = 'def build_integration_steps():\n    # MID: light integration (steps=3) — partly diffeomorphic, some folding removed.\n    return 3'

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 25,
        "end_line": 27,
        "content": _CONTENT,
    },
]
