"""Weak baseline for cv-count-architecture: PLAIN single-column CNN.

A shallow single-column fully-convolutional counter: one 3x3 conv stack, small (3x3)
receptive field, stride-8 density tail. Single filter size cannot cover the wide range
of object scales / crowding in the scene, and the small receptive field under-counts
dense, occluded regions -> the WORST counting MAE of the three architectures.

Literature anchor: the single-column CNN ablation in MCNN (Zhang et al., CVPR 2016,
Fig. 6) is the weakest, ShanghaiTech Part A MAE ~141 (best single column) vs the full
multi-column 110.2 vs CSRNet 68.2.
"""


def build_counter():
    import torch.nn as nn
    import torch.nn.functional as F

    def conv(cin, cout, k=3, d=1):
        return nn.Conv2d(cin, cout, k, padding=((k - 1) // 2) * d, dilation=d)

    class PlainCNN(nn.Module):
        """Single 3x3 column, small receptive field, few channels, stride 8."""

        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(2)
            self.c1 = nn.Sequential(conv(3, 16), nn.ReLU(True))
            self.c2 = nn.Sequential(conv(16, 24), nn.ReLU(True))
            self.c3 = nn.Sequential(conv(24, 24), nn.ReLU(True))
            self.tail = nn.Sequential(conv(24, 24), nn.ReLU(True), nn.Conv2d(24, 1, 1))

        def forward(self, x):
            x = self.pool(self.c1(x))
            x = self.pool(self.c2(x))
            x = self.pool(self.c3(x))
            return F.softplus(self.tail(x)).squeeze(1)   # (B,h,w) density

    return PlainCNN()
