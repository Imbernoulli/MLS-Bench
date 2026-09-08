"""GroupNorm normalization baseline for 2D CNNs.

Applies Group Normalization over channels for 2D feature maps.
Divides channels into groups and normalizes within each group,
making it independent of batch size.

Reference: Wu & He, "Group Normalization", ECCV 2018.

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_norm.py"

_CONTENT = """\
class CustomNorm(nn.Module):
    \"\"\"Group Normalization for 2D feature maps. Drop-in replacement for BatchNorm2d.

    Divides channels into groups and normalizes within each group independently.
    Works well with small batch sizes where BatchNorm statistics are noisy.

    Reference: Wu & He, "Group Normalization" (ECCV 2018)
    \"\"\"

    def __init__(self, num_features):
        super().__init__()
        num_groups = min(32, num_features)
        # Ensure num_features is divisible by num_groups
        while num_features % num_groups != 0:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups, num_features)

    def forward(self, x):
        return self.norm(x)
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 31,
        "end_line": 45,
        "content": _CONTENT,
    },
]
