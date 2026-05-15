"""RMSNorm baseline (basic).

Replaces LayerNorm with RMSNorm (Root Mean Square Normalization).
Simpler and faster — no mean subtraction, only scale by RMS.
Block structure unchanged (Pre-LN).

Reference: Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019)
Used in LLaMA, Gemma, etc.

Ops ordered bottom-to-top for line stability.
"""

_FILE = "nanoGPT/custom_pretrain.py"

_RMSNORM = """\
class LayerNorm(nn.Module):
    \"\"\"RMSNorm — Root Mean Square Layer Normalization.\"\"\"
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = 1e-5

    def forward(self, input):
        rms = input.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (input * rms).type_as(input) * self.weight
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 22,
        "end_line": 31,
        "content": _RMSNORM,
    },
]
