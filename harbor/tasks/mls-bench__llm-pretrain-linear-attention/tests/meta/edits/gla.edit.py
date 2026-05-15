"""Gated Linear Attention (GLA) baseline.

Replaces standard softmax attention with GLA from flash-linear-attention.
GLA uses gated recurrence with data-dependent decay for linear-complexity attention.
Sets use_pos_emb=False since linear attention doesn't use positional biases.

Reference: Yang et al., "Gated Linear Attention Transformers with Hardware-Efficient Training"

Ops ordered bottom-to-top for line stability.
"""

_FILE = "nanoGPT/custom_pretrain.py"

_GLA_ATTENTION = """\
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        from fla.layers import GatedLinearAttention
        self.attn = GatedLinearAttention(
            mode='chunk',
            hidden_size=config.n_embd,
            num_heads=config.n_head,
            expand_k=0.5,
            expand_v=1.0,
            use_output_gate=True,
            gate_fn='swish',
        )
        self.use_pos_emb = False

    @torch.compiler.disable
    def _attn_forward(self, x):
        return self.attn(x)

    def forward(self, x):
        o, _, _ = self._attn_forward(x)
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
        "content": _GLA_ATTENTION,
    },
]
