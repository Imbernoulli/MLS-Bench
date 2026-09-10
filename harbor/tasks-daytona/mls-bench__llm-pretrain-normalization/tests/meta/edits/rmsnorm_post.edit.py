"""RMSNorm + Sandwich-Norm baseline (medium).

Replaces LayerNorm with RMSNorm AND changes block structure from
Pre-LN to Sandwich-LN (normalize both before sublayer and after
residual addition).

Sandwich-Norm combines Pre-LN stability with Post-LN's advantage of
placing normalization after the residual, preventing representation
collapse in deep models. Reference:
  Ding et al., "CogView: Mastering Text-to-Image Generation via Transformers" (2021)

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

_SANDWICH_BLOCK = """\
class Block(nn.Module):
    \"\"\"Sandwich-Norm: Pre-LN + Post-LN with RMSNorm (CogView style).\"\"\"
    def __init__(self, config):
        super().__init__()
        self.ln_pre1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_post1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_pre2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.ln_post2 = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x):
        x = x + self.ln_post1(self.attn(self.ln_pre1(x)))
        x = x + self.ln_post2(self.mlp(self.ln_pre2(x)))
        return x
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 88,
        "end_line": 100,
        "content": _SANDWICH_BLOCK,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 22,
        "end_line": 31,
        "content": _RMSNORM,
    },
]
