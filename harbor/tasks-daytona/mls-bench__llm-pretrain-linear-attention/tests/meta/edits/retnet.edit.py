"""Multi-Scale Retention (RetNet) baseline.

Replaces standard softmax attention with MultiScaleRetention from flash-linear-attention.
RetNet uses exponential decay for recurrent attention with multi-scale heads.
Sets use_pos_emb=False since retention doesn't use positional embeddings.

Reference: Sun et al., "Retentive Network: A Successor to Transformer for Large Language Models"

Ops ordered bottom-to-top for line stability.
"""

_FILE = "nanoGPT/custom_pretrain.py"

_RETNET_ATTENTION = """\
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        from fla.layers import MultiScaleRetention
        self.attn = MultiScaleRetention(
            hidden_size=config.n_embd,
            num_heads=config.n_head,
            expand_k=1.0,
            expand_v=1.0,
            use_output_gate=True,
            gate_fn='swish',
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
        "content": _RETNET_ATTENTION,
    },
]
