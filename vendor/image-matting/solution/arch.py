"""Agent-editable surface: the WHOLE MATTING NETWORK (arch).

Return a torch.nn.Module `net` via build_net(in_ch) whose forward
    net(x, image=<B,3,H,W>, trimap=<B,H,W>) -> alpha (B,H,W) in [0,1]
takes x = concat(RGB, trimap-encoding) with `in_ch` channels and predicts a soft
alpha matte at FULL resolution. (image / trimap kwargs are provided for optional
guided refinement; a net may ignore them.) The trimap encoding (raw channel), loss
(fixed alpha-L1 + composition), data, optimiser, iterations, seed and eval are
FIXED; only the network changes. Scored by SAD (LOWER is better) in the trimap
UNKNOWN band, gmean over three trimap-width settings.

This is the STRICT-BAR direction: the ordering
    copy-trimap / constant (degenerate)  <  plain encoder-decoder
      <  DIM deep-matting (encoder-decoder + U-Net skips + a refinement stage,
         Xu et al. 2017 = SOTA)
must hold across ALL THREE settings.

The DEFAULT below is a deliberately weak PLAIN ENCODER-DECODER: it downsamples then
bilinearly upsamples the deepest feature back to full resolution with NO skip
connections and NO refinement, so it loses the fine transition detail -> high SAD.
Redesign the network (add U-Net skip connections that inject the encoder's high-res
features, and a second refinement stage as in Deep Image Matting) to recover a sharp
matte with clear headroom. A malformed / crashing net falls back to the harness
strong U-Net.
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
    # Default: PLAIN encoder-decoder. Encode to a stride-8 bottleneck, then bilinearly
    # upsample the deepest feature straight back to full resolution -> NO skip
    # connections, NO refinement -> the fine soft transition is lost -> high SAD.
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
