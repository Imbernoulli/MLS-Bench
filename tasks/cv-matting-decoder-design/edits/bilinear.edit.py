"""Weak baseline (negative control) for cv-matting-decoder-design: bilinear decoder.

Uses ONLY the deepest (stride-8) encoder feature, projects it to alpha and bilinearly
upsamples -> no skip connections, discards all high-resolution detail -> blurry matte,
high SAD. This is the starting default in vendor/image-matting/solution/decoder.py.
Reference: vendor/image-matting/baselines/decoder_bilinear.py
"""

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
    {"op": "replace", "file": _FILE, "start_line": 53, "end_line": 70, "content": _CONTENT},
]
