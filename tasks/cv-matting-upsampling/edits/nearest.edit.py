"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "image-matting/solution/upsampling.py"

_CONTENT = '''def build_upsampler(cin):
    class NearestUp(nn.Module):
        def forward(self, x):
            return F.interpolate(x, scale_factor=2, mode="nearest")
    return NearestUp()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 15, "end_line": 20, "content": _CONTENT},
]
