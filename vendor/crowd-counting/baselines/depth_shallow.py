"""Weak baseline for cv-count-depth: SHALLOW backbone.

A shallow feature extractor (one conv per pooling stage) has too little capacity to
resolve heavily crowded, occluded scenes -> it under-counts dense crowds -> higher
counting MAE at a fixed step budget.
"""


def build_deep_backbone():
    import torch.nn as nn

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
