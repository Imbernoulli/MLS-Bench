"""SiLU/Swish activation baseline.

Sigmoid Linear Unit: x * sigmoid(x). Self-gated, smooth, non-monotonic.

Reference: Ramachandran et al., "Searching for Activation Functions" (2017)
"""

_FILE = "pytorch-vision/custom_activation.py"

_CONTENT = """\
class CustomActivation(nn.Module):
    \"\"\"SiLU/Swish activation function.

    SiLU(x) = x * sigmoid(x).
    Self-gated activation discovered via automated search.
    \"\"\"

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.silu(x)
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
