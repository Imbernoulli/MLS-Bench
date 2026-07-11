"""Unmeasured full-protocol anchor candidate.

Retained for worker-side calibration. No score, ordering, or full-protocol runtime
is asserted before a complete 256x256, 100,000-step run.
"""

_FILE = "image-inpainting/solution/dilation.py"

_CONTENT = '''    class ModerateDilation(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(ch, ch, 3, 1, padding=1, dilation=1)
            self.c2 = nn.Conv2d(ch, ch, 3, 1, padding=2, dilation=2)
            self.act = nn.ReLU(True)

        def forward(self, x):
            x = self.act(self.c1(x))
            x = self.act(self.c2(x))
            return x

    return ModerateDilation()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 17, "end_line": 26, "content": _CONTENT},
]
