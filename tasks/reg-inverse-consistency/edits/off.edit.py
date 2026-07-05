"""Baseline for reg-inverse-consistency: STRONG: no inverse-consistency (inverse_w=0) — plain one-directional VoxelMorph, best alignment at this scale.
Reference: vendor/deformable-registration/baselines/*.py
"""

_FILE = "deformable-registration/solution/inverse.py"

_CONTENT = 'def build_inverse_weight():\n    # STRONG: no inverse-consistency (inverse_w=0) — plain one-directional VoxelMorph, best alignment at this scale.\n    return 0.0'

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 23,
        "end_line": 25,
        "content": _CONTENT,
    },
]
