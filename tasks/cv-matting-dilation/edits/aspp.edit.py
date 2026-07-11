"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "image-matting/solution/dilation.py"

_CONTENT = '''def build_dilation(ch):
    class ASPP(nn.Module):
        def __init__(self):
            super().__init__()
            self.d1 = nn.Conv2d(ch, ch, 3, padding=1, dilation=1)
            self.d2 = nn.Conv2d(ch, ch, 3, padding=2, dilation=2)
            self.d4 = nn.Conv2d(ch, ch, 3, padding=4, dilation=4)
            self.fuse = nn.Conv2d(3 * ch, ch, 1)
            self.bn = nn.BatchNorm2d(ch)

        def forward(self, x):
            y = torch.cat([self.d1(x), self.d2(x), self.d4(x)], 1)
            return x + F.relu(self.bn(self.fuse(y)))
    return ASPP()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 16, "end_line": 26, "content": _CONTENT},
]
