"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "image-matting/solution/decoder.py"

_CONTENT = '''def build_decoder(enc_channels):
    c0, c1, c2, c3 = enc_channels

    def cbr(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(True),
                             nn.Conv2d(b, b, 3, padding=1), nn.ReLU(True))

    class Dec(nn.Module):
        def __init__(self):
            super().__init__()
            self.up3 = cbr(c3 + c2, c2)
            self.up2 = cbr(c2 + c1, c1)
            self.up1 = cbr(c1 + c0, c0)
            self.out = nn.Conv2d(c0, 1, 1)

        def _u(self, x, r):
            return F.interpolate(x, size=r.shape[-2:], mode="bilinear", align_corners=False)

        def forward(self, feats):
            e0, e1, e2, e3 = feats
            d = self.up3(torch.cat([self._u(e3, e2), e2], 1))
            d = self.up2(torch.cat([self._u(d, e1), e1], 1))
            d = self.up1(torch.cat([self._u(d, e0), e0], 1))
            return torch.sigmoid(self.out(d)).squeeze(1)
    return Dec()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 17, "end_line": 33, "content": _CONTENT},
]
