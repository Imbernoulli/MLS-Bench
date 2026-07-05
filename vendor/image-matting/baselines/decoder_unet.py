"""Good baseline for cv-matting-decoder-design: U-Net skip-connection decoder.

Injects the encoder's high-resolution, low-level features into the decoder via SKIP
connections so the matte keeps sharp boundaries (U-Net; cf. IndexNet index-guided
upsampling, Lu et al. 2019) -> lower SAD and gradient error with clear headroom over
the bilinear decoder. Reference: vendor/image-matting/baselines/decoder_unet.py
"""


def build_decoder(enc_channels):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
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
    return Dec()
