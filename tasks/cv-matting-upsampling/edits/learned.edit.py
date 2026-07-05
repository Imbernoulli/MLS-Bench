"""SOTA baseline for cv-matting-upsampling: LEARNED upsample (bilinear + refine conv).

Bilinear upsample (x2) followed by a learned 3x3 refine conv (a lightweight learned
upsampler). Reconstructs a smooth, sharp soft edge instead of a blocky nearest-
neighbour tiling -> lowest SAD / gradient error with clear headroom.
Reference: vendor/image-matting/baselines/upsampling_learned.py
"""

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
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 38, "content": _CONTENT},
]
