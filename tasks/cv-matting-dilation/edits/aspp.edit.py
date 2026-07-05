"""SOTA baseline for cv-matting-dilation: DILATED multi-rate (ASPP-style) block.

Parallel dilated convs at rates 1/2/4 (atrous spatial pyramid pooling, Chen et al.
2017; dilated context, Iizuka et al. 2017) fused by a 1x1, added residually. Enlarges
the receptive field WITHOUT losing resolution, aggregating wide context across the
unknown band -> lowest SAD with clear headroom over a single 3x3.
Reference: vendor/image-matting/baselines/dilation_aspp.py
"""

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
    {"op": "replace", "file": _FILE, "start_line": 35, "end_line": 47, "content": _CONTENT},
]
