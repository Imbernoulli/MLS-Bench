"""Monocular-3D FEATURE-REPRESENTATION design surface (agent-editable) for mono3d-feature-representation.

Build the shared embedding the task heads consume, from the appearance crop and the geometry
feature vector. The geometry features carry the 2D-box pixel HEIGHT h2d — the key inverse-depth
cue for Z=f*H/h2d — plus box position/size; the crop carries physical-size and pose signal.

You implement:

    def build_feature_fusion(feat_dim, crop_hw) -> (module: nn.Module, forward: callable):

`forward(feat, crop) -> emb [B, 128]` produces the shared 128-d embedding. `feat` [B, feat_dim]
is the geometry feature vector; `crop` [B,3,crop_hw,crop_hw] the appearance crop.

The DEFAULT below is the WEAK baseline: an APPEARANCE-ONLY encoder that ignores the geometry
feature vector, discarding the h2d inverse-depth cue -> a much weaker embedding. FUSING appearance
+ geometry (see `common.RegionEncoder`) is far stronger. Everything else is fixed.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ================================================================
# EDITABLE REGION — build_feature_fusion(feat_dim, crop_hw) -> (module, forward)
# ================================================================
def build_feature_fusion(feat_dim, crop_hw):
    # WEAK DEFAULT: APPEARANCE-ONLY — encode the crop, IGNORE the geometry feature vector.
    class _AppOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            self.head = nn.Sequential(nn.Linear(48, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())

        def forward(self, feat, crop):
            return self.head(self.cnn(crop))

    mod = _AppOnly()

    def forward(feat, crop):
        return mod(feat, crop)

    return mod, forward
# ================================================================
# END EDITABLE REGION
# ================================================================
