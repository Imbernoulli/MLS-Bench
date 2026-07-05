"""SOTA baseline for cv-matting-attention: NON-LOCAL SELF-ATTENTION bottleneck.

A non-local self-attention block (contextual attention, Yu et al. 2018): each
bottleneck location attends to all others via a softmax over query-key similarity,
aggregating GLOBAL context so the unknown band can borrow appearance/structure from
anywhere in the image -> lowest SAD with clear headroom. Residual-added to preserve
the local feature. Reference: vendor/image-matting/baselines/attention_selfattn.py
"""

_FILE = "image-matting/solution/attention.py"

_CONTENT = '''def build_attention(ch):
    class SelfAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Conv2d(ch, ch // 4, 1)
            self.k = nn.Conv2d(ch, ch // 4, 1)
            self.v = nn.Conv2d(ch, ch, 1)
            self.gamma = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            b, c, h, w = x.shape
            q = self.q(x).view(b, -1, h * w).permute(0, 2, 1)   # (B,N,c')
            k = self.k(x).view(b, -1, h * w)                     # (B,c',N)
            attn = torch.softmax(torch.bmm(q, k) / (k.shape[1] ** 0.5), dim=-1)
            v = self.v(x).view(b, c, h * w)                      # (B,c,N)
            o = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, h, w)
            return x + self.gamma * o
    return SelfAttn()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 33, "end_line": 46, "content": _CONTENT},
]
