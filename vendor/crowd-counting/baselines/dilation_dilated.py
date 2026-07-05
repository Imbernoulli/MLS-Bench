"""Good baseline for cv-count-dilation: DILATED large-receptive-field block (CSRNet).

Dilated 3x3 convs (rate 2) enlarge the receptive field WITHOUT reducing resolution, so
the block aggregates large-scale context while keeping a dense, high-quality density
map -> lower counting MAE, especially in crowded regions. This is CSRNet's core idea
(Li et al., CVPR 2018).
"""


def build_backbone_block(cin):
    import torch.nn as nn

    def conv(ci, co, d):
        return nn.Conv2d(ci, co, 3, padding=d, dilation=d)

    class DilatedBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                conv(cin, 64, 2), nn.ReLU(True),
                conv(64, 64, 2), nn.ReLU(True),
                conv(64, 64, 2), nn.ReLU(True))
            self.out_channels = 64

        def forward(self, x):
            return self.net(x)

    return DilatedBlock()
