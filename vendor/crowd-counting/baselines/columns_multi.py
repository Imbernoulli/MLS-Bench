"""Good baseline for cv-count-columns: MULTI-column frontend (MCNN-style).

Three parallel columns with DIFFERENT filter sizes (9x9 / 7x7 / 5x5), concatenated.
The different receptive fields absorb the wide range of object scales / crowding, so it
counts multi-scale scenes better than a single column -> lower counting MAE. This is
MCNN's multi-column design (Zhang et al., CVPR 2016).
"""


def build_frontend():
    import torch
    import torch.nn as nn

    def column(k):
        p = (k - 1) // 2
        return nn.Sequential(
            nn.Conv2d(3, 20, k, padding=p), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(20, 40, k, padding=p), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(40, 40, k, padding=p), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(40, 24, k, padding=p), nn.ReLU(True))

    class MultiColumn(nn.Module):
        def __init__(self):
            super().__init__()
            self.large = column(9); self.medium = column(7); self.small = column(5)
            self.out_channels = 24 * 3

        def forward(self, x):
            return torch.cat([self.large(x), self.medium(x), self.small(x)], dim=1)

    return MultiColumn()
