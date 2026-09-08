"""Attention-based global pooling baseline.

Learns a spatial attention map from channel-mean activations using a
learnable inverse-temperature parameter.  When the temperature is near
zero the attention is uniform and the layer degrades to global average
pooling, providing a safe starting point for training.

Reference: Inspired by SENet attention (Hu et al., CVPR 2018) and
CBAM spatial attention (Woo et al., ECCV 2018), adapted for global pooling.

Ops ordered bottom-to-top for line stability.
"""

_FILE = "pytorch-vision/custom_pool.py"

_CONTENT = """\
class CustomPool(nn.Module):
    \"\"\"Learned attention-based global pooling.

    Computes a shared spatial attention map from the channel-mean activation,
    then performs an attention-weighted spatial average.  A learnable
    temperature parameter controls attention sharpness.

    The design is channel-agnostic (only 1 learnable scalar), so no lazy
    initialisation is needed and the parameter is always visible to the
    optimizer.  When temperature ~ 0 the attention is uniform and the layer
    reduces to global average pooling, providing a safe starting point.
    \"\"\"

    def __init__(self):
        super().__init__()
        # Learnable inverse-temperature; init near 0 => uniform attention => avg pool
        self.inv_temp = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # Shared spatial attention from channel-mean activation
        energy = x.mean(dim=1, keepdim=True)          # [B, 1, H, W]
        attn = torch.sigmoid(self.inv_temp * energy)   # [B, 1, H, W]
        # Attention-weighted spatial average (broadcast over C)
        pooled = (x * attn).sum(dim=(2, 3)) / (attn.sum(dim=(2, 3)) + 1e-8)
        return pooled  # [B, C]
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
