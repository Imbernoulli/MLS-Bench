"""Average + Max (AvgMax) Pooling baseline.

Element-wise average of global average pooling and global max pooling,
combining both summary statistics of spatial feature maps.

Reference: Common practice in image retrieval and classification.

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_pool.py"

_CONTENT = """\
class CustomPool(nn.Module):
    \"\"\"Average + Max Pooling.

    Element-wise mean of global average pooling and global max pooling.
    Combines mean-field statistics with peak activations.
    \"\"\"

    def __init__(self):
        super().__init__()

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        return ((avg + mx) / 2).view(x.size(0), -1)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 31,
        "end_line": 48,
        "content": _CONTENT,
    },
]
