"""ScaleNorm baseline for 2D CNNs.

Per-channel normalization using L2 norm over spatial dimensions with a
learned scale parameter. Simple, batch-independent, and stable for deep
networks due to its unit-norm property.

The filename is retained for compatibility with the task's baseline registry.

ScaleNorm: g * x / ||x||_spatial, where g is a learned per-channel scale.

Reference: Nguyen & Salazar, "Transformers without Tears: Improving the
Normalization of Self-Attention" (IWSLT 2019)
Adapted for 2D CNNs: L2 norm computed per-channel over (H, W).

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_norm.py"

_CONTENT = """\
class CustomNorm(nn.Module):
    \"\"\"ScaleNorm for 2D feature maps. Drop-in replacement for BatchNorm2d.

    Normalizes each channel of each sample to unit L2 norm over spatial
    dimensions, then applies a learned per-channel scale. Batch-independent
    and stable for very deep networks.

    Formula: scale * x / (||x||_{H,W} + eps)

    Reference: Nguyen & Salazar, IWSLT 2019 (adapted for 2D CNNs)
    \"\"\"

    def __init__(self, num_features):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_features) * 4.0)
        self.eps = 1e-5

    def forward(self, x):
        # x: [B, C, H, W]
        # L2 norm per channel per sample over spatial dims
        norm = x.norm(2, dim=(2, 3), keepdim=True) + self.eps
        x_normed = x / norm
        return self.scale.view(1, -1, 1, 1) * x_normed
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
