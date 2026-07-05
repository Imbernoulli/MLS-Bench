"""Weak baseline (negative control) for cv-matting-arch: PLAIN encoder-decoder.

Encodes to a stride-8 bottleneck then bilinearly upsamples the deepest feature
straight back to full resolution -> NO skip connections, NO refinement -> the fine
soft transition detail is lost -> high SAD. This is the starting default in
vendor/image-matting/solution/arch.py.
Reference: vendor/image-matting/baselines/arch_plain.py
"""

_FILE = "image-matting/solution/arch.py"

_CONTENT = '''def build_net(in_ch):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.e0 = _cbr(in_ch, 32)
            self.e1 = _cbr(32, 64)
            self.e2 = _cbr(64, 96)
            self.e3 = _cbr(96, 128)
            self.pool = nn.MaxPool2d(2)
            self.dec = nn.Sequential(_cbr(128, 64), nn.Conv2d(64, 1, 1))

        def forward(self, x, image=None, trimap=None):
            e0 = self.e0(x)
            e1 = self.e1(self.pool(e0))
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            a = self.dec(e3)
            a = F.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return torch.sigmoid(a).squeeze(1)
    return Net()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 42, "end_line": 64, "content": _CONTENT},
]
