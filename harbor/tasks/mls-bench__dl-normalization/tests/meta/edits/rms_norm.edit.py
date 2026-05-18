"""RMSNorm normalization baseline for 2D CNNs.

Applies Root Mean Square Layer Normalization over channels for 2D feature maps.
RMSNorm removes the mean centering of LayerNorm and only rescales by
the root-mean-square of the activations.

Reference: Zhang & Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019.

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_norm.py"

_CONTENT = """\
class CustomNorm(nn.Module):
    \"\"\"RMSNorm for 2D feature maps. Drop-in replacement for BatchNorm2d.

    Normalizes each sample by the root-mean-square of its channel activations,
    without mean centering. Reshapes [B,C,H,W] to [B,H,W,C] for per-channel
    RMS normalization, then reshapes back.

    Reference: Zhang & Sennrich, "Root Mean Square Layer Normalization" (NeurIPS 2019)
    \"\"\"

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        # x: [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x / rms * self.weight
        # [B, H, W, C] -> [B, C, H, W]
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
