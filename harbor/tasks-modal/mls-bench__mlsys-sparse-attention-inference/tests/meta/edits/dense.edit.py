"""Dense attention oracle baseline (density = 1.0).

Reference upper bound — runs the unmodified pretrained model's attention
exactly. This is the only baseline allowed to exceed the density budget,
which is signaled by the test_cmds setting ALLOW_DENSE_FLAG=1.
"""

_FILE = "sparse-attn-eval/custom_sparse_attn.py"

_CONTENT = """\

class SparseAttention(nn.Module):
    \"\"\"Dense attention oracle. Reports true full-attention density.

    The point of this baseline is to give an upper-bound quality reference;
    it is not meant to satisfy a real sparsity constraint.
    \"\"\"

    def __init__(self, head_dim, num_heads, block_size=64, density_budget=0.25):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.density_budget = density_budget
        self.last_density = None

    def forward(self, q, k, v, is_causal=False, scale=None):
        B, H, N, D = q.shape
        scale = scale if scale is not None else 1.0 / math.sqrt(D)
        # Use PyTorch's fused SDPA for efficient dense attention.
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0,
            is_causal=is_causal, scale=scale,
        )
        self.last_density = 1.0
        return out
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 31,
        "end_line": 103,
        "content": _CONTENT,
    },
]
