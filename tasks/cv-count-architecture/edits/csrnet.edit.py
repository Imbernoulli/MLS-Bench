"""SOTA baseline for cv-count-architecture: CSRNet-style DILATED backbone.

VGG-lite stem (3 poolings, stride 8) + a back-end of DILATED 3x3 convs (rate 2) that
enlarge the receptive field WITHOUT reducing resolution -> the LOWEST counting MAE.
Literature anchor: CSRNet ShanghaiTech Part A MAE 68.2 / Part B 10.6 (Li et al., CVPR
2018) -- strictly better than MCNN (110.2 / 26.4) and the plain single column (~141
Part A). Reference: vendor/crowd-counting/baselines/arch_csrnet.py
"""

_FILE = "crowd-counting/solution/arch.py"

_CONTENT = '''    def conv(ci, co, k=3, d=1):
        return nn.Conv2d(ci, co, k, padding=((k - 1) // 2) * d, dilation=d)

    class CSRNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(2)
            self.b1 = nn.Sequential(conv(3, 32), nn.ReLU(True), conv(32, 32), nn.ReLU(True))
            self.b2 = nn.Sequential(conv(32, 64), nn.ReLU(True), conv(64, 64), nn.ReLU(True))
            self.b3 = nn.Sequential(conv(64, 64), nn.ReLU(True), conv(64, 64), nn.ReLU(True))
            self.backend = nn.Sequential(
                conv(64, 64, 3, d=2), nn.ReLU(True),
                conv(64, 64, 3, d=2), nn.ReLU(True),
                conv(64, 32, 3, d=2), nn.ReLU(True))
            self.out = nn.Conv2d(32, 1, 1)

        def forward(self, x):
            x = self.pool(self.b1(x)); x = self.pool(self.b2(x)); x = self.pool(self.b3(x))
            return F.softplus(self.out(self.backend(x))).squeeze(1)   # (B,h,w) density
    return CSRNet()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 55, "end_line": 74, "content": _CONTENT},
]
