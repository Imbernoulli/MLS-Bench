"""SOTA baseline for cv-count-architecture: CSRNet-style DILATED backbone.

A VGG-style front-end with only THREE poolings (stride 8) followed by a back-end of
DILATED 3x3 convs (dilation rate 2). Dilation ENLARGES the receptive field WITHOUT
further reducing resolution, so the network aggregates large-scale context while
keeping a dense, high-quality stride-8 density map -> the LOWEST counting MAE of the
three architectures (the founding CSRNet result).

Literature anchor: CSRNet (Li et al., CVPR 2018) ShanghaiTech Part A MAE 68.2 /
Part B 10.6 -- strictly better than MCNN (110.2 / 26.4) and the plain single column
(~141 Part A). Dilated back-end (config "B", rate 2) is the winning configuration.
"""


def build_counter():
    import torch.nn as nn
    import torch.nn.functional as F

    def conv(cin, cout, k=3, d=1):
        return nn.Conv2d(cin, cout, k, padding=((k - 1) // 2) * d, dilation=d)

    class CSRNet(nn.Module):
        """VGG-lite frontend (stride 8) + dilated back-end (rate 2)."""

        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(2)
            self.b1 = nn.Sequential(conv(3, 32), nn.ReLU(True), conv(32, 32), nn.ReLU(True))
            self.b2 = nn.Sequential(conv(32, 64), nn.ReLU(True), conv(64, 64), nn.ReLU(True))
            self.b3 = nn.Sequential(conv(64, 64), nn.ReLU(True), conv(64, 64), nn.ReLU(True))
            # dilated back-end: large receptive field, resolution preserved
            self.backend = nn.Sequential(
                conv(64, 64, 3, d=2), nn.ReLU(True),
                conv(64, 64, 3, d=2), nn.ReLU(True),
                conv(64, 32, 3, d=2), nn.ReLU(True))
            self.out = nn.Conv2d(32, 1, 1)

        def forward(self, x):
            x = self.pool(self.b1(x))
            x = self.pool(self.b2(x))
            x = self.pool(self.b3(x))
            x = self.backend(x)
            return F.softplus(self.out(x)).squeeze(1)   # (B,h,w) density

    return CSRNet()
