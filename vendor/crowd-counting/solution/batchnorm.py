"""Agent-editable surface: NORMALIZATION in the backbone (none vs BatchNorm).

Define `build_backbone()` -> a torch.nn.Module mapping an image `(B, 3, H, W)` to
stride-8 features `(B, C, h, w)` (with a `.out_channels` attribute = C). A default
density head is attached after it; only the backbone (its normalization) changes.

WITHOUT normalization, the activation statistics drift across the wide count range
within a batch of crowded images, so optimisation is less stable and the density
calibration is noisier at a fixed step budget -> higher counting MAE. Adding BatchNorm
after each conv stabilises the statistics, converges better, and calibrates the density
-> lower MAE. This mirrors the CSRNet-with-BN (VGG16-BN) variant used for batched
crowd-counting training.

    def build_backbone():
        import torch.nn as nn
        def cbr(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1),
                                 nn.BatchNorm2d(cout), nn.ReLU(True))
        class BNBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.MaxPool2d(2)
                self.b1 = nn.Sequential(cbr(3,32), cbr(32,32))
                self.b2 = nn.Sequential(cbr(32,64), cbr(64,64))
                self.b3 = nn.Sequential(cbr(64,64), cbr(64,64))
                self.out_channels = 64
            def forward(self, x):
                x=self.pool(self.b1(x)); x=self.pool(self.b2(x)); x=self.pool(self.b3(x)); return x
        return BNBackbone()

The DEFAULT below is the deliberately weak NO-normalization backbone. A crashing /
malformed backbone falls back to the default fixed VGG-lite frontend.
"""
from __future__ import annotations

import torch.nn as nn


# ================================================================
# EDITABLE REGION — design the (normalization of the) backbone below
# ================================================================
def build_backbone():
    # Default: NO normalization (weak). Activation stats drift across the count range ->
    # less stable optimisation at fixed steps -> higher MAE.
    def conv(cin, cout, d=1):
        return nn.Conv2d(cin, cout, 3, padding=d, dilation=d)

    class PlainBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(2)
            self.b1 = nn.Sequential(conv(3, 32), nn.ReLU(True), conv(32, 32), nn.ReLU(True))
            self.b2 = nn.Sequential(conv(32, 64), nn.ReLU(True), conv(64, 64), nn.ReLU(True))
            self.b3 = nn.Sequential(conv(64, 64), nn.ReLU(True), conv(64, 64), nn.ReLU(True))
            self.out_channels = 64

        def forward(self, x):
            x = self.pool(self.b1(x)); x = self.pool(self.b2(x)); x = self.pool(self.b3(x))
            return x
    return PlainBackbone()
# ================================================================
# END EDITABLE REGION
# ================================================================
