"""Mamba2 baseline.

Replaces standard softmax attention with Mamba2 from flash-linear-attention.
Mamba2 uses selective state spaces with structured state space duality (SSD)
for hardware-efficient linear-time sequence modeling.
Sets use_pos_emb=False since Mamba2 uses convolutions for local context.

Reference: Dao & Gu, "Transformers are SSMs: Generalized Models and Efficient
           Algorithms Through Structured State Space Duality"

Ops ordered bottom-to-top for line stability.
"""

_FILE = "nanoGPT/custom_pretrain.py"

_MAMBA_ATTENTION = """\
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Mock selective_scan_cuda to prevent ABI mismatch crash on import
        import types, sys
        if 'selective_scan_cuda' not in sys.modules:
            sys.modules['selective_scan_cuda'] = types.ModuleType('selective_scan_cuda')
        from fla.layers import Mamba2
        # Mamba2 num_heads = expand * hidden_size / head_dim (default expand=2, head_dim=64)
        # NOT the same as transformer attention heads (config.n_head)
        self.attn = Mamba2(
            hidden_size=config.n_embd,
            num_heads=config.n_embd * 2 // 64,
        )
        self.use_pos_emb = False

    def forward(self, x):
        o, _, _ = self.attn(x)
        return o
"""

_BLOCK = """\
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 88,
        "end_line": 100,
        "content": _BLOCK,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 33,
        "end_line": 70,
        "content": _MAMBA_ATTENTION,
    },
]
