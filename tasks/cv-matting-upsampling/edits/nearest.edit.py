"""Weak baseline for cv-matting-upsampling: NEAREST-NEIGHBOUR upsample.

Replicates pixels (x2 nearest) -> blocky matte whose soft transition is aliased ->
higher SAD / gradient error. This is the starting default in
vendor/image-matting/solution/upsampling.py.
Reference: vendor/image-matting/baselines/upsampling_nearest.py
"""

_FILE = "image-matting/solution/upsampling.py"

_CONTENT = '''def build_upsampler(cin):
    class NearestUp(nn.Module):
        def forward(self, x):
            return F.interpolate(x, scale_factor=2, mode="nearest")
    return NearestUp()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 31, "end_line": 38, "content": _CONTENT},
]
