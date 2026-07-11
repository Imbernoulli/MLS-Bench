"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "image-matting/solution/attention.py"

_CONTENT = '''def build_attention(ch):
    class SqueezeExcitation(nn.Module):
        def __init__(self):
            super().__init__()
            reduced_channels = max(ch // 8, 1)
            self.reduce = nn.Conv2d(ch, reduced_channels, 1)
            self.expand = nn.Conv2d(reduced_channels, ch, 1)

        def forward(self, x):
            pooled = x.mean(dim=(-2, -1), keepdim=True)
            return torch.sigmoid(self.expand(F.relu(self.reduce(pooled))))
    return SqueezeExcitation()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 16, "end_line": 21, "content": _CONTENT},
]
