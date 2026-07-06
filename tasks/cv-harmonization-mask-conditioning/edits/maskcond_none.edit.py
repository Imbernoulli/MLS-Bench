"""cv-harmonization-mask-conditioning baseline: none (mask-BLIND (region-agnostic) — weak).
Reference: vendor/image-harmonization/baselines/maskcond_none.py
"""

_FILE = "image-harmonization/solution/maskcond.py"

_CONTENT = 'def get_mask_conditioning():\n    return "none"'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 24, "end_line": 26, "content": _CONTENT},
]
