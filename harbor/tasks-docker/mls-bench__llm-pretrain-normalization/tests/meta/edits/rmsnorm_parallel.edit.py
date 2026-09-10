"""RMSNorm + Parallel Attention/MLP baseline (strongest).

Replaces LayerNorm with RMSNorm AND changes block structure to
parallel attention + MLP (GPT-J / PaLM style). Both sublayers
operate on the same normalized input and their outputs are summed.

Reference: Wang & Komatsuzaki, "GPT-J-6B" (2021); Chowdhery et al., "PaLM" (2022)
Inspired by modded-nanogpt parallel residual streams.

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

_PARALLEL_BLOCK = """\
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        # Parallel: single norm, attention and MLP operate in parallel
        h = self.ln(x)
        x = x + self.attn(h) + self.mlp(h)
        return x
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 88,
        "end_line": 100,
        "content": _PARALLEL_BLOCK,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 22,
        "end_line": 31,
        "content": _RMSNORM,
    },
]
