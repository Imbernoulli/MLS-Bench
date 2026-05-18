"""Pre-activation ResBlock baseline (He et al., 2016 v2).

BN-ReLU-Conv order instead of Conv-BN-ReLU. Provides cleaner gradient flow
through identity shortcuts, especially beneficial for very deep networks.

Reference: He et al., "Identity Mappings in Deep Residual Networks" (ECCV 2016)
"""

_FILE = "pytorch-vision/custom_residual.py"

_CONTENT = """\
class CustomBlock(nn.Module):
    \"\"\"Pre-activation residual block (He et al., 2016 v2).

    Uses BN-ReLU-Conv order for cleaner gradient flow.
    Both main branch and shortcut share the same pre-activation.
    Residual scaling (alpha, init=0.1) stabilizes very deep networks.
    \"\"\"
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        pre = F.relu(self.bn1(x))
        out = self.conv1(pre)
        out = self.conv2(F.relu(self.bn2(out)))
        shortcut = self.downsample(x) if self.downsample is not None else x
        return shortcut + self.alpha * out
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
