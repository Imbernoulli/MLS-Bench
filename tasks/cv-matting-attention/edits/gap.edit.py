"""Weak baseline for cv-matting-attention: GLOBAL-AVERAGE-POOL bottleneck.

Replaces every spatial location with the channel-wise global mean (broadcast back),
destroying all spatial context at the bottleneck -> the unknown band cannot localise
the matte -> high SAD. This is the starting default in
vendor/image-matting/solution/attention.py.
Reference: vendor/image-matting/baselines/attention_gap.py
"""

_FILE = "image-matting/solution/attention.py"

_CONTENT = '''def build_attention(ch):
    class GAPBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(ch, ch, 1)

        def forward(self, x):
            g = x.mean(dim=(-2, -1), keepdim=True)
            g = self.proj(g)
            return g.expand_as(x)
    return GAPBlock()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 33, "end_line": 46, "content": _CONTENT},
]
