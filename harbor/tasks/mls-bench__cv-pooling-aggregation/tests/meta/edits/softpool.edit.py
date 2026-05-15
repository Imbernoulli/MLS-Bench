"""SoftPool-inspired global pooling baseline.

Uses softmax-weighted average pooling where the weights are derived from
the activations themselves. Higher activations contribute more, providing
a differentiable approximation to max pooling that retains more detail
than average pooling.

Reference: Stergiou et al., "Refining activation downsampling with SoftPool"
(ICCV 2021)

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_pool.py"

_CONTENT = """\
class CustomPool(nn.Module):
    \"\"\"SoftPool-inspired global pooling.

    Computes softmax-weighted spatial average where attention weights come
    from the activation magnitudes. Higher activations receive more weight,
    providing a smooth interpolation between avg and max pooling.

    Reference: Stergiou et al., ICCV 2021 (adapted for global pooling).
    \"\"\"

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # Compute per-channel spatial softmax weights
        flat = x.view(B, C, -1)  # [B, C, H*W]
        weights = F.softmax(flat, dim=2)  # softmax over spatial
        # Weighted average
        pooled = (flat * weights).sum(dim=2)  # [B, C]
        return pooled
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
