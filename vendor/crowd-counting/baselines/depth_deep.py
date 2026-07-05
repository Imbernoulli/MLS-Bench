"""Good baseline for cv-count-depth: DEEP backbone.

A deeper feature extractor (two convs per pooling stage + a post-pool refinement block)
has the capacity to resolve heavily crowded, occluded scenes -> it counts dense crowds
more accurately -> lower counting MAE. Depth is the standard lever behind VGG-16-based
crowd counters (CSRNet uses a 13-layer VGG front-end).
"""


def build_deep_backbone():
    import torch.nn as nn

    def cbr(ci, co):
        return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.ReLU(True))

    class Deep(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(2)
            self.b1 = nn.Sequential(cbr(3, 32), cbr(32, 32))
            self.b2 = nn.Sequential(cbr(32, 64), cbr(64, 64))
            self.b3 = nn.Sequential(cbr(64, 64), cbr(64, 64))
            self.refine = nn.Sequential(cbr(64, 64), cbr(64, 64))
            self.out_channels = 64

        def forward(self, x):
            x = self.pool(self.b1(x)); x = self.pool(self.b2(x)); x = self.pool(self.b3(x))
            return self.refine(x)

    return Deep()
