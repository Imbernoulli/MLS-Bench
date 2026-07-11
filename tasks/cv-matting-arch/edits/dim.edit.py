"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

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
            # U-Net decoder with SKIP connections
            self.up3 = _cbr(128 + 96, 96)
            self.up2 = _cbr(96 + 64, 64)
            self.up1 = _cbr(64 + 32, 32)
            self.out = nn.Conv2d(32, 1, 1)
            # SECOND refinement stage (residual): [image(3)+coarse alpha(1)] -> residual
            self.ref = nn.Sequential(
                nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(True),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(True),
                nn.Conv2d(32, 1, 3, padding=1))

        def _u(self, x, r):
            return F.interpolate(x, size=r.shape[-2:], mode="bilinear", align_corners=False)

        def forward(self, x, image=None, trimap=None):
            e0 = self.e0(x)
            e1 = self.e1(self.pool(e0))
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            d = self.up3(torch.cat([self._u(e3, e2), e2], 1))
            d = self.up2(torch.cat([self._u(d, e1), e1], 1))
            d = self.up1(torch.cat([self._u(d, e0), e0], 1))
            coarse = torch.sigmoid(self.out(d))                 # (B,1,H,W)
            if image is None:
                image = x[:, :3]
            r = self.ref(torch.cat([image, coarse], 1))         # residual
            fine = torch.sigmoid(self.out(d) + r)               # refined alpha
            return fine.squeeze(1)
    return Net()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 23, "end_line": 43, "content": _CONTENT},
]
