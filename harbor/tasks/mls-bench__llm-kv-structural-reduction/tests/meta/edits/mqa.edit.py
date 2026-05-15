"""Multi-query attention baseline for KV structural reduction.

Based on the MQA family introduced in "Fast Transformer Decoding: One Write-Head
is All You Need" (Shazeer, 2019).
"""

_FILE = "nanoGPT/custom_pretrain.py"

_MQA_REGION = """\
def build_kv_heads(config):
    \"\"\"Use one shared KV head for all query heads.\"\"\"

    n_kv_head = 1
    head_dim = config.n_embd // config.n_head
    return n_kv_head, head_dim


def cross_layer_share(layer_idx, config):
    return False


def latent_kv_project(k, v, config):
    return k, v, 1.0


def expand_kv_to_q_heads(tensor, target_heads):
    current_heads = tensor.size(1)
    if current_heads == target_heads:
        return tensor
    full_repeats = target_heads // current_heads
    remainder = target_heads % current_heads
    parts = []
    if full_repeats > 0:
        parts.append(tensor.repeat_interleave(full_repeats, dim=1))
    if remainder > 0:
        parts.append(tensor[:, :remainder, :, :])
    return torch.cat(parts, dim=1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.layer_idx = layer_idx
        self.n_kv_head, self.head_dim = build_kv_heads(config)
        self.share_across_layers = False

        q_dim = config.n_embd
        kv_dim = 2 * self.n_kv_head * self.head_dim
        self.c_attn = nn.Linear(config.n_embd, q_dim + kv_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )
        self.use_pos_emb = True
        self.head_sharing_ratio = float(self.n_head)

    def forward(self, x):
        bsz, seq_len, channels = x.size()
        qkv = self.c_attn(x)
        q, kv = qkv.split(
            [self.n_embd, 2 * self.n_kv_head * self.head_dim],
            dim=2,
        )
        k, v = kv.chunk(2, dim=2)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        k = expand_kv_to_q_heads(k, self.n_head)
        v = expand_kv_to_q_heads(v, self.n_head)
        self._last_latent_rank_ratio = 1.0
        self._last_kv_storage_ratio = 1.0
        self._uses_latent_compression = False
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        y = self.resid_dropout(self.c_proj(y))
        return y
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 36,
        "end_line": 155,
        "content": _MQA_REGION,
    },
]
