"""Mid baseline for cv-count-architecture: MULTI-COLUMN CNN (MCNN).

Three parallel columns with DIFFERENT filter sizes (large 9x9 / medium 7x7 / small
5x5), fused by a 1x1 conv, in the spirit of MCNN (Zhang et al., CVPR 2016). The
different receptive fields absorb the wide range of object scales / crowding, so it
counts noticeably better than a single-column plain CNN -> lower MAE.

Literature anchor: MCNN full multi-column ShanghaiTech Part A MAE 110.2 / Part B 26.4,
strictly better than the single-column ablation (~141 Part A) and worse than CSRNet
(68.2 Part A). Same stride-8 output, density counted by its integral.
"""


def build_counter():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def col(k):
        p = (k - 1) // 2
        return nn.Sequential(
            nn.Conv2d(3, 20, k, padding=p), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(20, 40, k, padding=p), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(40, 40, k, padding=p), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(40, 24, k, padding=p), nn.ReLU(True))

    class MCNN(nn.Module):
        """Multi-column: 9x9 / 7x7 / 5x5 columns fused by 1x1 conv + density tail."""

        def __init__(self):
            super().__init__()
            self.large = col(9)
            self.medium = col(7)
            self.small = col(5)
            self.fuse = nn.Sequential(
                nn.Conv2d(24 * 3, 64, 1), nn.ReLU(True),
                nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(True),
                nn.Conv2d(32, 1, 1))

        def forward(self, x):
            f = torch.cat([self.large(x), self.medium(x), self.small(x)], dim=1)
            return F.softplus(self.fuse(f)).squeeze(1)   # (B,h,w) density

    return MCNN()
