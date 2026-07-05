"""Weak baseline for cv-count-batchnorm: NO normalization backbone.

A plain VGG-lite backbone with no BatchNorm. With a batch of 8 crowded images the
activation statistics drift across the wide count range, so optimisation is less stable
and the density calibration is noisier -> higher counting MAE at a fixed step budget.
This is the original MCNN/CSRNet (no-BN) configuration.
"""


def build_backbone():
    import torch.nn as nn

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
