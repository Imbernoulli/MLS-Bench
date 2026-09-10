"""Squared ReLU activation baseline.

SquaredReLU(x) = ReLU(x)^2, scaled for variance control.

Adapted for CNN use: a small linear leak (0.1 * ReLU) is added so that
the gradient at x=0 is non-zero.  Pure ReLU(x)^2 has d/dx = 2*relu(x)
which vanishes at x=0; depthwise separable convolutions in MobileNetV2
(9 params per filter, expansion ratio 6) cannot learn through this dead
zone, causing training collapse to random chance.  The linear term gives
a constant gradient floor of 0.1*scale at x=0+.

Reference: So et al., "Primer: Searching for Efficient Transformers for
Language Modeling", NeurIPS 2021.
"""

_FILE = "pytorch-vision/custom_activation.py"

_CONTENT = """\
class CustomActivation(nn.Module):
    \"\"\"Squared ReLU activation function (CNN-adapted).

    SquaredReLU(x) = (ReLU(x)^2 + 0.1 * ReLU(x)) * 0.25.
    A small linear component keeps gradients alive at x=0 so that
    depthwise-separable convolutions can learn. Clamp at 4.0 prevents
    variance blow-up.

    Reference: So et al., "Primer: Searching for Efficient Transformers" (NeurIPS 2021)
    \"\"\"

    def __init__(self):
        super().__init__()

    def forward(self, x):
        r = F.relu(x).clamp(max=4.0)
        return (r * r + 0.1 * r) * 0.25
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 32,
        "end_line": 49,
        "content": _CONTENT,
    },
]
