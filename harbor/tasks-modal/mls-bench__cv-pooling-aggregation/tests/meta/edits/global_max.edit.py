"""Global Max Pooling baseline.

Replaces adaptive average pooling with adaptive max pooling, selecting the
maximum activation per channel across spatial dimensions.

Reference: Standard practice in fine-grained recognition and retrieval.

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_pool.py"

_CONTENT = """\
class CustomPool(nn.Module):
    \"\"\"Global Max Pooling.

    Selects the maximum activation per channel across spatial dimensions.
    Captures the most salient features rather than averaging over all positions.
    \"\"\"

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)
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
