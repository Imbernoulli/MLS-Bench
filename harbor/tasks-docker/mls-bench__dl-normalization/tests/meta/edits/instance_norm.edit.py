"""InstanceNorm normalization baseline for 2D CNNs.

Applies Instance Normalization over each channel independently for 2D feature maps.
Normalizes over (H, W) for each channel in each sample.

Reference: Ulyanov et al., "Instance Normalization: The Missing Ingredient for
Fast Stylization", 2016.

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_norm.py"

_CONTENT = """\
class CustomNorm(nn.Module):
    \"\"\"Instance Normalization for 2D feature maps. Drop-in replacement for BatchNorm2d.

    Normalizes each channel independently over spatial dimensions (H, W).
    Uses affine=True to include learnable scale and shift parameters.

    Reference: Ulyanov et al., "Instance Normalization" (2016)
    \"\"\"

    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=True)

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
