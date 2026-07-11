"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "image-matting/solution/upsampling.py"

_CONTENT = '''def build_upsampler(cin):
    class LearnedUp(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(cin, cin, 3, padding=1)

        def forward(self, x):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            return F.relu(self.conv(x))
    return LearnedUp()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 15, "end_line": 20, "content": _CONTENT},
]
