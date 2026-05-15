"""DeltaNet baseline.

Replaces standard softmax attention with DeltaNet from flash-linear-attention.
DeltaNet uses delta update rules with short convolutions for linear attention.
Sets use_pos_emb=False since DeltaNet handles sequence ordering internally.

Reference: Schlag et al., "Linear Transformers Are Secretly Fast Weight Programmers"
           Yang et al., "Parallelizing Linear Transformers with the Delta Rule over Sequence Length"

Ops ordered bottom-to-top for line stability.
"""

_FILE = "nanoGPT/custom_pretrain.py"

_DELTANET_ATTENTION = """\
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        from fla.layers import DeltaNet
        self.attn = DeltaNet(
            hidden_size=config.n_embd,
            num_heads=config.n_head,
            use_beta=True,
            use_short_conv=True,
            conv_size=4,
            qk_activation='silu',
            qk_norm='l2',
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
        "content": _DELTANET_ATTENTION,
    },
]
