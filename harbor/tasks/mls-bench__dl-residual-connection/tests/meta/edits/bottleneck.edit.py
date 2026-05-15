"""Bottleneck residual block baseline (He et al., 2016).

1x1 conv (reduce) -> 3x3 conv -> 1x1 conv (expand) with expansion=4.
Reduces computation in the 3x3 conv while increasing output channels.

Reference: He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)
"""

_FILE = "pytorch-vision/custom_residual.py"

_CONTENT = """\
class CustomBlock(nn.Module):
    \"\"\"Bottleneck residual block with expansion=4.\"\"\"
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
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
