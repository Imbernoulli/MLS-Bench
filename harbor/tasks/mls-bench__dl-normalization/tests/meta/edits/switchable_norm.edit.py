"""Switchable Normalization baseline for 2D CNNs.

Learns to switch between BatchNorm, InstanceNorm, and LayerNorm via
per-channel importance weights. Each normalization computes its own
statistics, and the final output is a weighted combination.

Reference: Luo et al., "Differentiable Learning-to-Normalize via
Switchable Normalization", ICLR 2019.

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_norm.py"

_CONTENT = """\
class CustomNorm(nn.Module):
    \"\"\"Switchable Normalization for 2D feature maps. Drop-in replacement for BatchNorm2d.

    Learns to combine BatchNorm, InstanceNorm, and LayerNorm statistics via
    softmax-weighted importance weights. Adapts normalization strategy per
    channel during training.

    Reference: Luo et al., "Differentiable Learning-to-Normalize via
    Switchable Normalization" (ICLR 2019)
    \"\"\"

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.eps = 1e-5
        # Learnable affine parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        # Importance weights for mean (3 norms) and var (3 norms)
        self.mean_weight = nn.Parameter(torch.ones(3))
        self.var_weight = nn.Parameter(torch.ones(3))

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # Softmax over importance weights
        mean_w = F.softmax(self.mean_weight, dim=0)
        var_w = F.softmax(self.var_weight, dim=0)
        # Instance stats: per (B, C) over (H, W)
        mean_in = x.mean(dim=(2, 3), keepdim=True)
        var_in = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        # Layer stats: per B over (C, H, W)
        mean_ln = x.mean(dim=(1, 2, 3), keepdim=True)
        var_ln = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        # Batch stats: per C over (B, H, W)
        mean_bn = x.mean(dim=(0, 2, 3), keepdim=True)
        var_bn = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
        # Weighted combination
        mean = mean_w[0] * mean_in + mean_w[1] * mean_ln + mean_w[2] * mean_bn
        var = var_w[0] * var_in + var_w[1] * var_ln + var_w[2] * var_bn
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
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
