"""RWKV-6 baseline.

Replaces standard softmax attention with RWKV6Attention from flash-linear-attention.
RWKV-6 uses data-dependent linear recurrence with LoRA-parametrized decay,
combining token-mixing and gated output for efficient linear-time sequence modeling.
Sets use_pos_emb=False since RWKV handles sequence ordering via decay.

Reference: Peng et al., "Eagle and Finch: RWKV with Matrix-Valued States and
           Dynamic Recurrence"

Ops ordered bottom-to-top for line stability.
"""

_FILE = "nanoGPT/custom_pretrain.py"

_RWKV6_ATTENTION = """\
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Prevent transformers 5.5.3 from importing torchvision (ABI mismatch)
        import sys, types, importlib
        if 'torchvision' not in sys.modules:
            tv = types.ModuleType('torchvision')
            tv.__version__ = '0.0.0'
            tv.__spec__ = importlib.machinery.ModuleSpec('torchvision', None)
            tv.transforms = types.ModuleType('torchvision.transforms')
            tv.transforms.InterpolationMode = type('InterpolationMode', (), {})
            sys.modules['torchvision'] = tv
            sys.modules['torchvision.transforms'] = tv.transforms
        from fla.layers import RWKV6Attention
        self.attn = RWKV6Attention(
            mode='chunk',
            hidden_size=config.n_embd,
            num_heads=config.n_head,
            expand_k=1.0,
            expand_v=1.0,
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
        "content": _RWKV6_ATTENTION,
    },
]
