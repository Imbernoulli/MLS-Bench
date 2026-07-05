"""Weak baseline for cv-count-architecture: PLAIN single-column CNN.

One 3x3 filter size, small receptive field, stride-8 density tail. Cannot cover the wide
range of object scales / crowding -> highest counting MAE of the three architectures.
Literature anchor: MCNN single-column ablation ~ ShanghaiTech Part A MAE 141 (Zhang et
al., CVPR 2016). Reference: vendor/crowd-counting/baselines/arch_plain.py
"""

_FILE = "crowd-counting/solution/arch.py"

_CONTENT = '''    def conv(ci, co, k=3, d=1):
        return nn.Conv2d(ci, co, k, padding=((k - 1) // 2) * d, dilation=d)

    class PlainCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(2)
            self.c1 = nn.Sequential(conv(3, 16), nn.ReLU(True))
            self.c2 = nn.Sequential(conv(16, 24), nn.ReLU(True))
            self.c3 = nn.Sequential(conv(24, 24), nn.ReLU(True))
            self.tail = nn.Sequential(conv(24, 24), nn.ReLU(True), nn.Conv2d(24, 1, 1))

        def forward(self, x):
            x = self.pool(self.c1(x))
            x = self.pool(self.c2(x))
            x = self.pool(self.c3(x))
            return F.softplus(self.tail(x)).squeeze(1)   # (B,h,w) density
    return PlainCNN()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 55, "end_line": 74, "content": _CONTENT},
]
