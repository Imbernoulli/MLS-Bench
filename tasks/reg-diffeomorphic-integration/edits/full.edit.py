"""Baseline for reg-diffeomorphic-integration: STRONG: full integration (steps=7, VoxelMorph-diff) — fold-free diffeomorphism.
Reference: vendor/deformable-registration/baselines/*.py
"""

_FILE = "deformable-registration/solution/integration.py"

_CONTENT = 'def build_integration_steps():\n    # STRONG: full integration (steps=7, VoxelMorph-diff) — fold-free diffeomorphism.\n    return 7'

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 25,
        "end_line": 27,
        "content": _CONTENT,
    },
]
