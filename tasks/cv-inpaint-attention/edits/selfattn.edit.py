"""Unmeasured full-protocol anchor candidate.

Retained for worker-side calibration. No score, ordering, or full-protocol runtime
is asserted before a complete 256x256, 100,000-step run.
"""

_FILE = "image-inpainting/solution/attention.py"

_CONTENT = '''    class NonLocal(nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = nn.Conv2d(ch, ch // 4, 1)
            self.phi = nn.Conv2d(ch, ch // 4, 1)
            self.g = nn.Conv2d(ch, ch // 4, 1)
            self.o = nn.Conv2d(ch // 4, ch, 1)
            self.gamma = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            b, c, h, w = x.shape
            t = self.theta(x).view(b, -1, h * w).permute(0, 2, 1)   # (B, N, c')
            p = self.phi(x).view(b, -1, h * w)                      # (B, c', N)
            attn = torch.softmax(torch.bmm(t, p), dim=-1)           # (B, N, N)
            gg = self.g(x).view(b, -1, h * w).permute(0, 2, 1)      # (B, N, c')
            out = torch.bmm(attn, gg).permute(0, 2, 1).view(b, -1, h, w)
            return x + self.gamma * self.o(out)

    return NonLocal()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 17, "end_line": 21, "content": _CONTENT},
]
