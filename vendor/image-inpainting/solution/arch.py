"""Agent-editable full-resolution architecture surface for image inpainting."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# EDITABLE REGION
def build_net(in_ch):
    c = 64

    def norm(channels):
        return nn.GroupNorm(32, channels)

    class ResearchUNet(nn.Module):
        def __init__(self):
            super().__init__()
            b = 8 * c
            self.e1 = nn.Sequential(nn.Conv2d(in_ch, c, 5, 1, 2), norm(c), nn.ELU(True))
            self.e2 = nn.Sequential(nn.Conv2d(c, 2*c, 4, 2, 1), norm(2*c), nn.ELU(True))
            self.e3 = nn.Sequential(nn.Conv2d(2*c, 4*c, 4, 2, 1), norm(4*c), nn.ELU(True))
            self.e4 = nn.Sequential(nn.Conv2d(4*c, b, 4, 2, 1), norm(b), nn.ELU(True))
            self.context = nn.ModuleList([
                nn.Conv2d(b, b, 3, 1, dilation, dilation=dilation)
                for dilation in (1, 2, 4, 8)
            ])
            self.d3 = nn.Sequential(nn.Conv2d(b + 4*c, 4*c, 3, 1, 1), norm(4*c), nn.ELU(True))
            self.d2 = nn.Sequential(nn.Conv2d(4*c + 2*c, 2*c, 3, 1, 1), norm(2*c), nn.ELU(True))
            self.d1 = nn.Sequential(nn.Conv2d(2*c + c, c, 3, 1, 1), norm(c), nn.ELU(True))
            self.out = nn.Conv2d(c, 3, 3, 1, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(e1)
            e3 = self.e3(e2)
            value = self.e4(e3)
            for layer in self.context:
                value = F.elu(layer(value), inplace=True)
            value = self.d3(torch.cat([
                F.interpolate(value, scale_factor=2, mode="bilinear", align_corners=False), e3
            ], dim=1))
            value = self.d2(torch.cat([
                F.interpolate(value, scale_factor=2, mode="bilinear", align_corners=False), e2
            ], dim=1))
            value = self.d1(torch.cat([
                F.interpolate(value, scale_factor=2, mode="bilinear", align_corners=False), e1
            ], dim=1))
            return torch.sigmoid(self.out(value))

    return ResearchUNet()
# END EDITABLE REGION
