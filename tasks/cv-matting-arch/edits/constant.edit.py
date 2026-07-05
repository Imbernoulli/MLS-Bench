"""Degenerate baseline (negative control) for cv-matting-arch: CONSTANT / copy-trimap.

Ignores the encoder entirely and predicts a constant alpha derived from the trimap
(0.5 in the unknown band). Because the GT alpha in the unknown band is a genuine soft
ramp spanning 0->1 (mean != 0.5), this scores a large SAD (~ CONST_HALF_SAD) and
MAXIMAL gradient error -> it is beaten by every real matting net, confirming the
metric is monotone in matting quality.
Reference: vendor/image-matting/baselines/arch_constant.py
"""

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
    {"op": "replace", "file": _FILE, "start_line": 42, "end_line": 64, "content": _CONTENT},
]
