"""Agent-editable whole-network surface for full-inventory image matting.

Keep build_net(in_ch) and the documented forward contract. The selected module is
trained and evaluated directly; load, type, shape, range, or numerical failures
invalidate the run rather than selecting another implementation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cbr(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                         nn.ReLU(True), nn.Conv2d(cout, cout, 3, padding=1),
                         nn.BatchNorm2d(cout), nn.ReLU(True))


# ================================================================
# EDITABLE REGION — design the whole matting network below
# ================================================================
def build_net(in_ch):
    # Native plain encoder-decoder implementation.
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.e0 = _cbr(in_ch, 32)
            self.e1 = _cbr(32, 64)
            self.e2 = _cbr(64, 96)
            self.e3 = _cbr(96, 128)
            self.pool = nn.MaxPool2d(2)
            self.dec = nn.Sequential(_cbr(128, 64), nn.Conv2d(64, 1, 1))

        def forward(self, x, image=None, trimap=None):
            e0 = self.e0(x)
            e1 = self.e1(self.pool(e0))
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            a = self.dec(e3)
            a = F.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return torch.sigmoid(a).squeeze(1)
    return Net()
# ================================================================
# END EDITABLE REGION
# ================================================================
