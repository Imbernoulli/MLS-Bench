"""mono3d-feature-representation baseline: feature_fused.

Auto-generated from vendor/mono3d-detection/baselines/feature_fused.py. Replaces the editable region of
mono3d-detection/solution/feature_fusion.py (the `build_feature_fusion` surface) with the feature_fused implementation.
"""

_FILE = "mono3d-detection/solution/feature_fusion.py"

_CONTENT = 'def build_feature_fusion(feat_dim, crop_hw):\n    enc = common.RegionEncoder(feat_dim, crop_hw)\n\n    def forward(feat, crop):\n        return enc(feat, crop)\n\n    return enc, forward'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 27, "end_line": 49, "content": _CONTENT},
]
