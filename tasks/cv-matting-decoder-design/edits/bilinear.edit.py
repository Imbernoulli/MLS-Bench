"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "image-matting/solution/decoder.py"

_CONTENT = '''def build_decoder(enc_channels):
    # Default: deepest-feature-only bilinear-upsample decoder (no skip connections).
    c0, c1, c2, c3 = enc_channels

    class Dec(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(nn.Conv2d(c3, 32, 3, padding=1), nn.ReLU(True),
                                      nn.Conv2d(32, 1, 1))

        def forward(self, feats):
            a = self.proj(feats[-1])
            a = F.interpolate(a, size=feats[0].shape[-2:], mode="bilinear",
                              align_corners=False)
            return torch.sigmoid(a).squeeze(1)
    return Dec()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 17, "end_line": 33, "content": _CONTENT},
]
