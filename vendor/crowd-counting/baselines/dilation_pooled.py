"""Weak baseline for cv-count-dilation: POOLED small-receptive-field block.

The back-end block downsamples (extra pooling) and uses plain 3x3 convs: the receptive
field is small and the density map resolution is further reduced, so large-scale
context is lost and dense scenes are under-counted -> higher counting MAE. This is the
pre-CSRNet design that dilation was introduced to fix.
"""


def build_backbone_block(cin):
    import torch.nn as nn

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
