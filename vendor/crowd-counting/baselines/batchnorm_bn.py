"""Good baseline for cv-count-batchnorm: BatchNorm backbone.

The same VGG-lite backbone with BatchNorm after each conv. BN stabilises the activation
statistics across the wide count range within a batch, so training converges better at
a fixed step budget and the density is better calibrated -> lower counting MAE. Mirrors
the CSRNet-with-BN (VGG16-BN) variant used to enable batched crowd-counting training.
"""


def build_backbone():
    import torch.nn as nn

    def cbr(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1),
                             nn.BatchNorm2d(cout), nn.ReLU(True))

    class BNBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(2)
            self.b1 = nn.Sequential(cbr(3, 32), cbr(32, 32))
            self.b2 = nn.Sequential(cbr(32, 64), cbr(64, 64))
            self.b3 = nn.Sequential(cbr(64, 64), cbr(64, 64))
            self.out_channels = 64

        def forward(self, x):
            x = self.pool(self.b1(x)); x = self.pool(self.b2(x)); x = self.pool(self.b3(x))
            return x

    return BNBackbone()
