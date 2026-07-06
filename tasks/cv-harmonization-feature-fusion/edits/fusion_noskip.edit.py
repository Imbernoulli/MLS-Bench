"""cv-harmonization-feature-fusion baseline: noskip (no skip connections — weak).
Reference: vendor/image-harmonization/baselines/fusion_noskip.py
"""

_FILE = "image-harmonization/solution/fusion.py"

_CONTENT = "def get_fusion_config():\n    return {'skips': False}"

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 17, "end_line": 19, "content": _CONTENT},
]
