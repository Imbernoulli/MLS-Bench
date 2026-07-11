"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "image-matting/solution/dilation.py"

_CONTENT = '''def build_dilation(ch):
    class SingleConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(ch, ch, 3, padding=1)
            self.bn = nn.BatchNorm2d(ch)

        def forward(self, x):
            return x + F.relu(self.bn(self.conv(x)))
    return SingleConv()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 16, "end_line": 26, "content": _CONTENT},
]
