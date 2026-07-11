"""Unmeasured full-protocol reference candidate. It has no accepted ranking until fresh terminal measurements exist."""

_FILE = "image-matting/solution/arch.py"

_CONTENT = '''def build_net(in_ch):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(1))  # keep an optimisable param

        def forward(self, x, image=None, trimap=None):
            # copy the trimap value (0.5 in unknown) -> constant 0.5 matte in the band
            if trimap is not None:
                a = trimap.clone()
            else:
                a = torch.full(x.shape[:1] + x.shape[-2:], 0.5, device=x.device)
            return (a + 0.0 * self.bias).clamp(0, 1)
    return Net()'''

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 23, "end_line": 43, "content": _CONTENT},
]
