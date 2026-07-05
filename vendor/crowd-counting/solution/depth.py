"""Agent-editable surface: the BACKBONE DEPTH (shallow vs deep feature extractor).

Define `build_deep_backbone()` -> a torch.nn.Module mapping an image `(B, 3, H, W)` to
stride-8 features `(B, C, h, w)` (with a `.out_channels` attribute = C). A default
density head is attached after it; only the backbone depth changes.

A SHALLOW backbone (one conv per pooling stage) has too little capacity to resolve
heavily crowded, occluded scenes -> it under-counts dense crowds -> higher counting MAE.
A DEEPER backbone (two convs per stage + a post-pool refinement block) has the capacity
to disentangle overlapping objects -> lower MAE. Depth is the standard lever behind
VGG-16-based crowd counters (CSRNet uses a 13-layer VGG front-end).

    def build_deep_backbone():
        import torch.nn as nn
        def cbr(ci, co): return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.ReLU(True))
        class Deep(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.MaxPool2d(2)
                self.b1 = nn.Sequential(cbr(3,32), cbr(32,32))
                self.b2 = nn.Sequential(cbr(32,64), cbr(64,64))
                self.b3 = nn.Sequential(cbr(64,64), cbr(64,64))
                self.refine = nn.Sequential(cbr(64,64), cbr(64,64))
                self.out_channels = 64
            def forward(self, x):
                x=self.pool(self.b1(x)); x=self.pool(self.b2(x)); x=self.pool(self.b3(x)); return self.refine(x)
        return Deep()

The DEFAULT below is the deliberately weak SHALLOW backbone. A crashing / malformed
backbone falls back to the default fixed VGG-lite frontend.
"""
from __future__ import annotations

import torch.nn as nn


# ================================================================
# EDITABLE REGION — design the (depth of the) backbone below
# ================================================================
def build_deep_backbone():
    # Default: SHALLOW backbone (weak). One conv per stage -> too little capacity for
    # heavily crowded scenes -> higher MAE.
    class Shallow(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(2)
            self.b1 = nn.Sequential(nn.Conv2d(3, 24, 3, padding=1), nn.ReLU(True))
            self.b2 = nn.Sequential(nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(True))
            self.b3 = nn.Sequential(nn.Conv2d(48, 64, 3, padding=1), nn.ReLU(True))
            self.out_channels = 64

        def forward(self, x):
            x = self.pool(self.b1(x)); x = self.pool(self.b2(x)); x = self.pool(self.b3(x))
            return x
    return Shallow()
# ================================================================
# END EDITABLE REGION
# ================================================================
