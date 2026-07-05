"""Weak baseline (negative control) for cv-matting-decoder-design: bilinear decoder.

Uses ONLY the deepest (stride-8) encoder feature, projects it to alpha and bilinearly
upsamples 8x -> no skip connections, discards all high-resolution detail -> blurry
matte, high SAD / gradient error. This is the starting default in
vendor/image-matting/solution/decoder.py.
"""


def build_decoder(enc_channels):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
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
    return Dec()
