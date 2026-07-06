"""cv-harmonization-mask-conditioning baseline: concat (mask concat 4th channel (DoveNet) — strong).
Reference: vendor/image-harmonization/baselines/maskcond_concat.py
"""

_FILE = "image-harmonization/solution/maskcond.py"

_CONTENT = 'def get_mask_conditioning():\n    return "concat"'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 24, "end_line": 26, "content": _CONTENT},
]
