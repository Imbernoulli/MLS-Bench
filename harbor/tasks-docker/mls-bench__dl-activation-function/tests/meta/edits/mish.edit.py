"""Mish activation baseline.

Mish(x) = x * tanh(softplus(x)). Self-regularized, smooth, non-monotonic.

Reference: Misra, "Mish: A Self Regularized Non-Monotonic Activation Function" (2019)
"""

_FILE = "pytorch-vision/custom_activation.py"

_CONTENT = """\
class CustomActivation(nn.Module):
    \"\"\"Mish activation function.

    Mish(x) = x * tanh(softplus(x)).
    Self-regularized non-monotonic activation with smooth gradients.
    \"\"\"

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
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
