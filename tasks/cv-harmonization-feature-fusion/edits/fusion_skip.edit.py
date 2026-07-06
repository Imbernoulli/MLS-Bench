"""cv-harmonization-feature-fusion baseline: skip (U-Net skip connections — strong).
Reference: vendor/image-harmonization/baselines/fusion_skip.py
"""

_FILE = "image-harmonization/solution/fusion.py"

_CONTENT = "def get_fusion_config():\n    return {'skips': True}"

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 17, "end_line": 19, "content": _CONTENT},
]
