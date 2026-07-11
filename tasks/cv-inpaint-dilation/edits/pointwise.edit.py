"""Unmeasured full-protocol anchor candidate.

Retained for worker-side calibration. No score, ordering, or full-protocol runtime
is asserted before a complete 256x256, 100,000-step run.
"""

_FILE = "image-inpainting/solution/dilation.py"

_CONTENT = '''    class Pointwise(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv2d(ch, ch, 1)
            self.act = nn.ReLU(True)
            m = torch.zeros(1, ch, 1, 1)
            m[:, : ch // 2] = 1.0                 # keep only half the channels
            self.register_buffer("chmask", m)

        def forward(self, x):
            return self.act(self.c(x)) * self.chmask

    return Pointwise()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 17, "end_line": 26, "content": _CONTENT},
]
