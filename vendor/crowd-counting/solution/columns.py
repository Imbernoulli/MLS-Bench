"""Agent-editable surface: the FRONTEND columns (single vs multi-column).

Define `build_frontend()` -> a torch.nn.Module mapping an image `(B, 3, H, W)` to
stride-8 features `(B, C, h, w)` (with a `.out_channels` attribute = C). A default
density head is attached after it; only the frontend changes.

A SINGLE-column frontend uses one filter size / receptive field: it cannot cover the
wide range of object scales / crowding, so it mis-counts multi-scale scenes -> higher
counting MAE. A MULTI-column frontend (MCNN-style) runs parallel columns with DIFFERENT
filter sizes (9x9 / 7x7 / 5x5) and concatenates them, absorbing scale variation ->
lower MAE (Zhang et al., CVPR 2016).

    def build_frontend():
        import torch, torch.nn as nn
        def column(k):
            p = (k - 1) // 2
            return nn.Sequential(
                nn.Conv2d(3,16,k,padding=p), nn.ReLU(True), nn.MaxPool2d(2),
                nn.Conv2d(16,32,k,padding=p), nn.ReLU(True), nn.MaxPool2d(2),
                nn.Conv2d(32,32,k,padding=p), nn.ReLU(True), nn.MaxPool2d(2),
                nn.Conv2d(32,22,k,padding=p), nn.ReLU(True))
        class MultiColumn(nn.Module):
            def __init__(self):
                super().__init__()
                self.large=column(9); self.medium=column(7); self.small=column(5)
                self.out_channels = 22*3
            def forward(self, x):
                return torch.cat([self.large(x), self.medium(x), self.small(x)], dim=1)
        return MultiColumn()

The DEFAULT below is the deliberately weak SINGLE-column frontend. A crashing / malformed
frontend falls back to the default multi-column frontend.
"""
from __future__ import annotations

import torch.nn as nn


# ================================================================
# EDITABLE REGION — design the frontend (columns) below
# ================================================================
def build_frontend():
    # Default: SINGLE-column frontend (weak). One filter size / scale -> mis-counts
    # multi-scale scenes -> higher MAE.
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
# ================================================================
# END EDITABLE REGION
# ================================================================
