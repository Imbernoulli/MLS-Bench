"""Gated residual block baseline (inspired by learnable residual scaling).

Uses a learnable scalar gate to scale the residual branch before adding to the
shortcut, similar to how per-layer resid_lambdas scale residual streams in
transformer architectures. The gate is initialized near 1.0 so the block starts
close to a standard residual connection.

Reference: Inspired by learnable residual scaling; unlike ReZero, this
baseline initializes the gate at 1.0 to start from standard residual behavior.
"""

_FILE = "pytorch-vision/custom_residual.py"

_CONTENT = """\
class CustomBlock(nn.Module):
    \"\"\"Residual block with learnable residual gate (scalar scaling).

    A learnable parameter alpha scales the residual branch output before
    adding to the shortcut: out = shortcut(x) + alpha * F(x).
    Initialized at alpha=1.0 (standard residual behavior).
    \"\"\"
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
        # Learnable residual gate initialized at 1.0
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shortcut(x) + self.alpha * out
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
