"""cv-harmonization-mask-conditioning baseline: gated (concat + mask-gated output blend).
Reference: vendor/image-harmonization/baselines/maskcond_gated.py
"""

_FILE = "image-harmonization/solution/maskcond.py"

_CONTENT = 'def get_mask_conditioning():\n    return "gated"'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 24, "end_line": 26, "content": _CONTENT},
]
