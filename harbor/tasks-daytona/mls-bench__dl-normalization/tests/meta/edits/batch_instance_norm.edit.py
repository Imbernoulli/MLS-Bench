"""Batch-Instance Normalization baseline for 2D CNNs.

Interpolates between BatchNorm and InstanceNorm via a learnable gate
parameter rho per channel. When rho=1 it is pure BatchNorm; when rho=0
it is pure InstanceNorm.

Reference: Nam & Kim, "Batch-Instance Normalization for Adaptively
Style-Invariant Neural Networks", NeurIPS 2018.

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_norm.py"

_CONTENT = """\
class CustomNorm(nn.Module):
    \"\"\"Batch-Instance Normalization for 2D feature maps. Drop-in replacement for BatchNorm2d.

    Learns a per-channel gate rho in [0, 1] (via sigmoid) that interpolates
    between BatchNorm statistics and InstanceNorm statistics.

    Reference: Nam & Kim, "Batch-Instance Normalization for Adaptively
    Style-Invariant Neural Networks" (NeurIPS 2018)
    \"\"\"

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.eps = 1e-5
        # Learnable affine parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        # Gate parameter (before sigmoid); init at 1.0 -> sigmoid ~ 0.73 -> mostly BN
        self.rho = nn.Parameter(torch.ones(num_features) * 1.0)

    def forward(self, x):
        # x: [B, C, H, W]
        gate = torch.sigmoid(self.rho).view(1, -1, 1, 1)
        # Batch stats: per C over (B, H, W)
        mean_bn = x.mean(dim=(0, 2, 3), keepdim=True)
        var_bn = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
        # Instance stats: per (B, C) over (H, W)
        mean_in = x.mean(dim=(2, 3), keepdim=True)
        var_in = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        # Interpolate
        x_bn = (x - mean_bn) / (var_bn + self.eps).sqrt()
        x_in = (x - mean_in) / (var_in + self.eps).sqrt()
        x_norm = gate * x_bn + (1 - gate) * x_in
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
