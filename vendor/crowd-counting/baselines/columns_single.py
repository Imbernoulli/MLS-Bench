"""Weak baseline for cv-count-columns: SINGLE-column frontend.

One column with a single 3x3 filter size and a small receptive field. A single scale
cannot cover the wide range of object sizes / crowding, so it under-counts multi-scale
scenes -> higher counting MAE. This is the single-column ablation MCNN improves on.
"""


def build_frontend():
    import torch.nn as nn

    class SingleColumn(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(2)
            self.b1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(True),
                                    nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(True))
            self.b2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(True),
                                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True))
            self.b3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True),
                                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True))
            self.out_channels = 64

        def forward(self, x):
            x = self.pool(self.b1(x)); x = self.pool(self.b2(x)); x = self.pool(self.b3(x))
            return x

    return SingleColumn()
