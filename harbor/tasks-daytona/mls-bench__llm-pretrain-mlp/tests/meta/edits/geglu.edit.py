"""GeGLU MLP baseline (strongest).

Gated Linear Unit with GELU activation. Strong variant used in many
modern architectures (Gemma, etc.).
Hidden dim adjusted to 8/3 * n_embd to keep parameter count similar.

Reference: Shazeer, "GLU Variants Improve Transformer" (2020)
"""

_FILE = "nanoGPT/custom_pretrain.py"

_GEGLU_MLP = """\
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(8 / 3 * config.n_embd)
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # GeGLU: GELU(xW1) * (xW3) then project back
        return self.dropout(self.c_proj(F.gelu(self.w1(x)) * self.w3(x)))
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 72,
        "end_line": 86,
        "content": _GEGLU_MLP,
    },
]
