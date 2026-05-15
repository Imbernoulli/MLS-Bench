"""GELU activation baseline.

Gaussian Error Linear Unit -- smooth approximation to the ReLU that weights
inputs by their magnitude under a Gaussian CDF.

Reference: Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)" (2016)
"""

_FILE = "pytorch-vision/custom_activation.py"

_CONTENT = """\
class CustomActivation(nn.Module):
    \"\"\"GELU activation function.

    GELU(x) = x * Phi(x) where Phi is the Gaussian CDF.
    Smooth, non-monotonic, allows small negative values.
    \"\"\"

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)
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
