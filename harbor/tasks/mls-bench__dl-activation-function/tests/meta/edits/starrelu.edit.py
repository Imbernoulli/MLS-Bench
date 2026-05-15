"""Fixed StarReLU-style activation baseline.

StarReLU uses s * ReLU(x)^2 + b with learned scalars in the paper. This
CNN-adapted baseline fixes the scale and bias inside the activation formula.

Adapted for CNN use: a small linear leak (0.1 * ReLU) is added to the
squared term so that the gradient does not vanish at x=0.  Without this,
depthwise separable convolutions (e.g. MobileNetV2 inverted bottlenecks)
collapse to random-chance accuracy because their per-channel 3x3 filters
cannot learn through a purely quadratic dead zone.

Reference: Yu et al., "MetaFormer Baselines for Vision", TPAMI 2024
(originally arXiv 2022).
"""

_FILE = "pytorch-vision/custom_activation.py"

_CONTENT = """\
class CustomActivation(nn.Module):
    \"\"\"Fixed StarReLU-style activation function (CNN-adapted).

    StarReLU(x) = s * (ReLU(x)^2 + alpha * ReLU(x)) + b.
    A small linear component (alpha=0.1) keeps the gradient alive at x=0
    so that depthwise-separable convolutions (MobileNetV2) can learn.
    Clamp at 4.0 prevents variance blow-up in deep layers.

    Reference: Yu et al., "MetaFormer Baselines for Vision" (TPAMI 2024)
    \"\"\"

    def __init__(self):
        super().__init__()

    def forward(self, x):
        r = F.relu(x).clamp(max=4.0)
        return 0.5 * (r * r + 0.1 * r - 0.5)
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
