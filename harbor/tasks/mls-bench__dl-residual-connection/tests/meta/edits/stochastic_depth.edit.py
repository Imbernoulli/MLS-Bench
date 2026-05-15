"""Stochastic depth baseline (Huang et al., 2016).

Randomly drops entire residual blocks during training with a linear-decay
survival probability. Early blocks are almost always kept while deeper blocks
are dropped more frequently, acting as an implicit ensemble regularizer.
At test time, the residual output is deterministically scaled by the survival
probability. Especially effective for very deep networks (ResNet-110+).

Reference: Huang et al., "Deep Networks with Stochastic Depth" (ECCV 2016)
"""

_FILE = "pytorch-vision/custom_residual.py"

_CONTENT = """\
class CustomBlock(nn.Module):
    \"\"\"Residual block with stochastic depth (Huang et al., 2016).

    During training, each block's residual branch is randomly dropped with
    probability (1 - survival_prob). The survival probability linearly decays
    from 1.0 (first block) to p_L (last block). At test time, the residual
    output is deterministically scaled by the survival probability.
    \"\"\"
    expansion = 1
    _block_counter = 0
    _p_last = 0.5  # survival prob of the deepest block

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
        # Reset counter when first block of a new model is created
        # (CIFAR ResNets always start layer1 with in_planes=16, planes=16, stride=1)
        if in_planes == 16 and planes == 16 and stride == 1:
            CustomBlock._block_counter = 0
        CustomBlock._block_counter += 1
        self.block_idx = CustomBlock._block_counter

    def forward(self, x):
        shortcut = self.shortcut(x)
        L = CustomBlock._block_counter
        p = 1.0 - (self.block_idx / L) * (1.0 - CustomBlock._p_last)
        if self.training:
            if torch.rand(1).item() < p:
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                return F.relu(out + shortcut)
            else:
                return F.relu(shortcut)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return F.relu(p * out + shortcut)
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
