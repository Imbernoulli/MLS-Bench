"""SwiGLU MLP baseline.

Replaces GELU MLP with SwiGLU (Gated Linear Unit with SiLU activation).
Used in LLaMA, PaLM, and other modern architectures.
Hidden dim adjusted to 8/3 * n_embd to keep parameter count similar.

Reference: Shazeer, "GLU Variants Improve Transformer" (2020)
"""

_FILE = "nanoGPT/custom_pretrain.py"

_SWIGLU_MLP = """\
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(8 / 3 * config.n_embd)
        # Round to nearest multiple of 64 for efficiency
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU: SiLU(xW1) * (xW3) then project back
        return self.dropout(self.c_proj(F.silu(self.w1(x)) * self.w3(x)))
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 72,
        "end_line": 86,
        "content": _SWIGLU_MLP,
    },
]
