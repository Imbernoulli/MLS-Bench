"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "image-matting/solution/attention.py"

_CONTENT = '''def build_attention(ch):
    class MeanGate(nn.Module):
        def forward(self, x):
            return torch.sigmoid(x.mean(dim=(-2, -1), keepdim=True))
    return MeanGate()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 16, "end_line": 21, "content": _CONTENT},
]
