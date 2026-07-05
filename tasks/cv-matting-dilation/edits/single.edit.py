"""Weak baseline for cv-matting-dilation: SINGLE plain 3x3 conv block.

A single 3x3 conv at the bottleneck (residual). Limited receptive field, no
multi-scale context aggregation across the wide unknown band -> higher SAD. This is
the starting default in vendor/image-matting/solution/dilation.py.
Reference: vendor/image-matting/baselines/dilation_single.py
"""

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
    {"op": "replace", "file": _FILE, "start_line": 35, "end_line": 47, "content": _CONTENT},
]
