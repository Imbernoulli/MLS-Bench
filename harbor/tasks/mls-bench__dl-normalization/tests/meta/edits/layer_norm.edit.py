"""LayerNorm normalization baseline.

Applies Layer Normalization across the channel dimension for 2D feature maps.
Reshapes [B,C,H,W] -> [B,H,W,C], applies LayerNorm(C), reshapes back.

Reference: Ba et al., "Layer Normalization" (2016)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_norm.py"

_CONTENT = """\
class CustomNorm(nn.Module):
    \"\"\"Layer Normalization for 2D feature maps. Drop-in replacement for BatchNorm2d.

    Reshapes [B,C,H,W] to [B,H,W,C], applies LayerNorm over channel dim,
    then reshapes back. Normalizes across channels for each spatial location.

    Reference: Ba et al., "Layer Normalization" (2016)
    \"\"\"

    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, H, W, C] -> LayerNorm -> [B, C, H, W]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
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
