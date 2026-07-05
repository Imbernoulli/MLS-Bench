"""mono3d-feature-representation baseline: feature_appearance_only.

Auto-generated from vendor/mono3d-detection/baselines/feature_appearance_only.py. Replaces the editable region of
mono3d-detection/solution/feature_fusion.py (the `build_feature_fusion` surface) with the feature_appearance_only implementation.
"""

_FILE = "mono3d-detection/solution/feature_fusion.py"

_CONTENT = 'def build_feature_fusion(feat_dim, crop_hw):\n    class _AppOnly(nn.Module):\n        def __init__(self):\n            super().__init__()\n            self.cnn = nn.Sequential(\n                nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),\n                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),\n                nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.ReLU(),\n                nn.AdaptiveAvgPool2d(1), nn.Flatten(),\n            )\n            self.head = nn.Sequential(nn.Linear(48, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())\n\n        def forward(self, feat, crop):\n            return self.head(self.cnn(crop))     # geometry features `feat` IGNORED\n\n    mod = _AppOnly()\n\n    def forward(feat, crop):\n        return mod(feat, crop)\n\n    return mod, forward'

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 27, "end_line": 49, "content": _CONTENT},
]
