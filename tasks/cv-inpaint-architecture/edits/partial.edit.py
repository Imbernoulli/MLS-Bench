"""Unmeasured full-protocol anchor candidate.

Retained for worker-side calibration. No score, ordering, or full-protocol runtime
is asserted before a complete 256x256, 100,000-step run.
"""

_FILE = "image-inpainting/solution/arch.py"

_CONTENT = '''    c = 64

    class PartialConv2d(nn.Module):
        def __init__(self, ic, oc, k, s, p):
            super().__init__()
            self.conv = nn.Conv2d(ic, oc, k, s, p, bias=True)
            self.register_buffer("wones", torch.ones(1, 1, k, k))
            self.s, self.p, self.win = s, p, float(k * k)

        def forward(self, x, mask):
            with torch.no_grad():
                valid = F.conv2d(mask[:, :1], self.wones, stride=self.s, padding=self.p)
                new_mask = (valid > 0).float()
                ratio = torch.clamp(self.win / (valid + 1e-6), max=2.0) * new_mask
            out = self.conv(x * mask)
            b = self.conv.bias.view(1, -1, 1, 1)
            out = (out - b) * ratio + b
            return out * new_mask, new_mask

    def _up(t):
        return F.interpolate(t, scale_factor=2, mode="nearest")

    class PConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            b = 8 * c
            self.act = nn.ReLU(True)
            self.e1 = PartialConv2d(3, c, 5, 1, 2)
            self.e2 = PartialConv2d(c, 2 * c, 4, 2, 1)
            self.e3 = PartialConv2d(2 * c, 4 * c, 4, 2, 1)
            self.e4 = PartialConv2d(4 * c, b, 4, 2, 1)
            self.mid = PartialConv2d(b, b, 3, 1, 1)
            self.n2 = nn.InstanceNorm2d(2 * c, affine=True)
            self.n3 = nn.InstanceNorm2d(4 * c, affine=True)
            self.n4 = nn.InstanceNorm2d(b, affine=True)
            self.d3 = PartialConv2d(b + 4 * c, 4 * c, 3, 1, 1)
            self.d2 = PartialConv2d(4 * c + 2 * c, 2 * c, 3, 1, 1)
            self.d1 = PartialConv2d(2 * c + c, c, 3, 1, 1)
            self.out = nn.Conv2d(c, 3, 3, 1, 1)

        def forward(self, x):
            rgb = x[:, :3]; valid = x[:, 3:4]
            e1, m1 = self.e1(rgb, valid); e1 = self.act(e1)
            e2, m2 = self.e2(e1, m1); e2 = self.act(self.n2(e2))
            e3, m3 = self.e3(e2, m2); e3 = self.act(self.n3(e3))
            e4, m4 = self.e4(e3, m3); e4 = self.act(self.n4(e4))
            mm, mmask = self.mid(e4, m4); mm = self.act(mm)
            u = torch.cat([_up(mm), e3], 1); um = torch.maximum(_up(mmask), m3)
            u, um = self.d3(u, um); u = self.act(u)
            u = torch.cat([_up(u), e2], 1); um = torch.maximum(_up(um), m2)
            u, um = self.d2(u, um); u = self.act(u)
            u = torch.cat([_up(u), e1], 1); um = torch.maximum(_up(um), m1)
            u, um = self.d1(u, um); u = self.act(u)
            return torch.sigmoid(self.out(u))

    return PConvNet()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 11, "end_line": 51, "content": _CONTENT},
]
