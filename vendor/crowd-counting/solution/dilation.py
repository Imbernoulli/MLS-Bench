"""Agent-editable surface: the BACKBONE BACK-END BLOCK (dilation / receptive field).

Define `build_backbone_block(cin)` -> a torch.nn.Module mapping stem features
`(B, cin, h, w)` to `(B, C, h, w)` (with a `.out_channels` attribute = C). It runs after
a FIXED VGG-lite stem; only this block changes.

A POOLED small-receptive-field block downsamples and uses plain 3x3 convs: the
receptive field stays small and resolution is lost, so large-scale context is missed
and dense scenes are under-counted -> higher counting MAE. A DILATED block (3x3 convs at
dilation rate 2) enlarges the receptive field WITHOUT reducing resolution, aggregating
large-scale context while keeping a dense density map -> lower MAE. This is CSRNet's
core idea (Li et al., CVPR 2018).

    def build_backbone_block(cin):
        import torch.nn as nn
        def conv(ci, co, d): return nn.Conv2d(ci, co, 3, padding=d, dilation=d)
        class DilatedBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(conv(cin,64,2), nn.ReLU(True),
                                         conv(64,64,2), nn.ReLU(True),
                                         conv(64,64,2), nn.ReLU(True))
                self.out_channels = 64
            def forward(self, x): return self.net(x)
        return DilatedBlock()

The DEFAULT below is the deliberately weak POOLED small-RF block. A crashing / malformed
block falls back to the default dilated block.
"""
from __future__ import annotations

import torch.nn as nn


# ================================================================
# EDITABLE REGION — design the backbone back-end block below
# ================================================================
def build_backbone_block(cin):
    # Default: POOLED small-receptive-field block (weak). Extra pooling + plain convs ->
    # small RF, lost resolution -> misses large-scale context -> higher MAE.
    class PooledBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(cin, 64, 3, padding=1), nn.ReLU(True),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            self.out_channels = 64

        def forward(self, x):
            return self.net(x)
    return PooledBlock()
# ================================================================
# END EDITABLE REGION
# ================================================================
