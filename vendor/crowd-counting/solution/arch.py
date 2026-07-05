"""Agent-editable surface: the FULL image->density crowd-counting ARCHITECTURE.

Return a torch.nn.Module `counter` mapping an input image batch `(B, 3, H, W)` to a
NON-NEGATIVE per-pixel DENSITY MAP `(B, h, w)`; the object count is its spatial
INTEGRAL (divided by the fixed DENSITY_SCALE=100 by the harness). You design the ENTIRE
backbone + density tail -- this is the core crowd-counting architecture decision.

The founding weak->strong->SOTA progression of density-map crowd counting:
  - PLAIN single-column CNN  (weak): one filter size, small receptive field. Cannot
    cover the wide range of object scales / crowding -> highest counting MAE.
    (MCNN single-column ablation ~ ShanghaiTech Part A MAE 141, Zhang et al. CVPR 2016.)
  - MULTI-COLUMN CNN (MCNN)  (mid): parallel columns with DIFFERENT filter sizes
    (9x9/7x7/5x5) fused by a 1x1 conv -> absorbs scale variation -> lower MAE.
    (MCNN Part A MAE 110.2 / Part B 26.4, Zhang et al. CVPR 2016.)
  - DILATED backbone (CSRNet) (SOTA): VGG-style stem (3 poolings, stride 8) + a back-end
    of DILATED convs (rate 2) that enlarge the receptive field WITHOUT losing
    resolution -> the lowest MAE. (CSRNet Part A MAE 68.2 / Part B 10.6, Li et al.
    CVPR 2018 -- strictly better than MCNN and the plain single column.)

    def build_counter():
        import torch.nn as nn, torch.nn.functional as F
        def conv(ci, co, k=3, d=1):
            return nn.Conv2d(ci, co, k, padding=((k-1)//2)*d, dilation=d)
        class CSRNet(nn.Module):                        # dilated SOTA backbone
            def __init__(self):
                super().__init__()
                self.pool = nn.MaxPool2d(2)
                self.b1 = nn.Sequential(conv(3,32), nn.ReLU(True), conv(32,32), nn.ReLU(True))
                self.b2 = nn.Sequential(conv(32,64), nn.ReLU(True), conv(64,64), nn.ReLU(True))
                self.b3 = nn.Sequential(conv(64,64), nn.ReLU(True), conv(64,64), nn.ReLU(True))
                self.backend = nn.Sequential(
                    conv(64,64,3,d=2), nn.ReLU(True), conv(64,64,3,d=2), nn.ReLU(True),
                    conv(64,32,3,d=2), nn.ReLU(True))
                self.out = nn.Conv2d(32, 1, 1)
            def forward(self, x):
                x = self.pool(self.b1(x)); x = self.pool(self.b2(x)); x = self.pool(self.b3(x))
                return F.softplus(self.out(self.backend(x))).squeeze(1)  # (B,h,w)
        return CSRNet()

The DEFAULT below is the deliberately weak PLAIN single-column CNN. Switching to the
multi-column (MCNN) design lowers the counting MAE, and the dilated (CSRNet) backbone
lowers it further -- the strongest architecture. A malformed / crashing counter falls
back to the default fixed VGG-lite frontend + density head.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# EDITABLE REGION — design the full image->density counter below
# ================================================================
def build_counter():
    # Default: PLAIN single-column CNN (weak). One 3x3 filter size, small receptive
    # field, stride-8 density tail -> cannot handle scale variation -> highest MAE.
    def conv(ci, co, k=3, d=1):
        return nn.Conv2d(ci, co, k, padding=((k - 1) // 2) * d, dilation=d)

    class PlainCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(2)
            self.c1 = nn.Sequential(conv(3, 24), nn.ReLU(True))
            self.c2 = nn.Sequential(conv(24, 32), nn.ReLU(True))
            self.c3 = nn.Sequential(conv(32, 32), nn.ReLU(True))
            self.tail = nn.Sequential(conv(32, 32), nn.ReLU(True), nn.Conv2d(32, 1, 1))

        def forward(self, x):
            x = self.pool(self.c1(x))
            x = self.pool(self.c2(x))
            x = self.pool(self.c3(x))
            return F.softplus(self.tail(x)).squeeze(1)   # (B,h,w) density
    return PlainCNN()
# ================================================================
# END EDITABLE REGION
# ================================================================
