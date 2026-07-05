"""mono3d-feature-representation WEAK baseline: APPEARANCE-ONLY encoder (drop geometry feats).

Encode ONLY the rendered appearance crop with the CNN and ZERO out the geometry feature vector
(the normalized 2D-box center/size, log h2d/w2d, aspect, focal). The single most important
monocular-3D cue is the 2D-box pixel HEIGHT h2d (it drives the geometry depth Z=f*H/h2d); dropping
the geometry features removes that cue from the embedding, so the encoder must guess depth/pose
from the tiny crop alone -> much weaker embedding, lower AP3D. Reference: every monocular-3D
detector fuses the box geometry with appearance; appearance alone underdetermines the 3D box.
"""
import torch
import torch.nn as nn


def build_feature_fusion(feat_dim, crop_hw):
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
            return self.head(self.cnn(crop))     # geometry features `feat` IGNORED

    mod = _AppOnly()

    def forward(feat, crop):
        return mod(feat, crop)

    return mod, forward
