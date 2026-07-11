"""Unmeasured full-protocol anchor candidate.

Retained for worker-side calibration. No score, ordering, or full-protocol runtime
is asserted before a complete 256x256, 100,000-step run.
"""

_FILE = "image-inpainting/solution/attention.py"

_CONTENT = '''    class GlobalPool(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Conv2d(ch, ch, 1)

        def forward(self, x):
            g = x.mean(dim=(2, 3), keepdim=True)          # (B,ch,1,1) global vector
            g = self.fc(g)
            return g.expand_as(x)                          # broadcast -> no spatial info

    return GlobalPool()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 17, "end_line": 21, "content": _CONTENT},
]
