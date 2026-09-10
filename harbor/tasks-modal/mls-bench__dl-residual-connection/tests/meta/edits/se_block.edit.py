"""Squeeze-and-Excitation ResBlock baseline (Hu et al., 2018).

Standard basic block with SE channel attention after the second conv+bn.
GAP -> FC(reduce) -> ReLU -> FC(expand) -> Sigmoid -> scale.

Reference: Hu et al., "Squeeze-and-Excitation Networks" (CVPR 2018)
"""

_FILE = "pytorch-vision/custom_residual.py"

_CONTENT = """\
class CustomBlock(nn.Module):
    \"\"\"Basic residual block with Squeeze-and-Excitation attention.\"\"\"
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )
        # Squeeze-and-Excitation
        reduction = 16
        mid = max(planes // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(planes, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, planes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # SE channel attention
        w = self.se(out).unsqueeze(-1).unsqueeze(-1)
        out = out * w
        out += self.shortcut(x)
        return F.relu(out)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 31,
        "end_line": 61,
        "content": _CONTENT,
    },
]
