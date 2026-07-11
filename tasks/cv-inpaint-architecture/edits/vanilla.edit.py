"""Unmeasured full-protocol anchor candidate.

Retained for worker-side calibration. No score, ordering, or full-protocol runtime
is asserted before a complete 256x256, 100,000-step run.
"""

_FILE = "image-inpainting/solution/arch.py"

_CONTENT = '''    c = 64

    class VanillaNet(nn.Module):
        def __init__(self):
            super().__init__()
            b = 8 * c
            self.act = nn.ReLU(True)
            self.e1 = nn.Conv2d(in_ch, c, 5, 1, 2)
            self.e2 = nn.Conv2d(c, 2 * c, 4, 2, 1)
            self.e3 = nn.Conv2d(2 * c, 4 * c, 4, 2, 1)
            self.e4 = nn.Conv2d(4 * c, b, 4, 2, 1)
            self.mid = nn.Conv2d(b, b, 3, 1, 1)
            self.d3 = nn.Conv2d(b + 4 * c, 4 * c, 3, 1, 1)
            self.d2 = nn.Conv2d(4 * c + 2 * c, 2 * c, 3, 1, 1)
            self.d1 = nn.Conv2d(2 * c + c, c, 3, 1, 1)
            self.out = nn.Conv2d(c, 3, 3, 1, 1)

        def _up(self, x):
            return F.interpolate(x, scale_factor=2, mode="nearest")

        def forward(self, x):
            e1 = self.act(self.e1(x)); e2 = self.act(self.e2(e1))
            e3 = self.act(self.e3(e2)); e4 = self.act(self.e4(e3))
            m = self.act(self.mid(e4))
            u = self.act(self.d3(torch.cat([self._up(m), e3], 1)))
            u = self.act(self.d2(torch.cat([self._up(u), e2], 1)))
            u = self.act(self.d1(torch.cat([self._up(u), e1], 1)))
            return torch.sigmoid(self.out(u))

    return VanillaNet()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 11, "end_line": 51, "content": _CONTENT},
]
