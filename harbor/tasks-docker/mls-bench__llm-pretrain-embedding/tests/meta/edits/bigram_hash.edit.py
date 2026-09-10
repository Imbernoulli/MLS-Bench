"""Bigram hash embedding baseline (strongest).

Augments token embeddings with a hash-based bigram embedding that captures
local token-pair context. Uses XOR hashing of consecutive token pairs into
a larger embedding table.

Reference: KellerJordan/modded-nanogpt record #62
"""

_FILE = "nanoGPT/custom_pretrain.py"

_BIGRAM_HASH_EMBEDDING = """\
class TokenEmbedding(nn.Module):
    \"\"\"Token + position + bigram hash embedding.\"\"\"
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.block_size = config.block_size
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size
        # Bigram hash embedding: 5x vocab for hash collision reduction
        self.bigram_vocab_size = config.vocab_size * 5
        self.bigram_embed = nn.Embedding(self.bigram_vocab_size, config.n_embd)
        nn.init.zeros_(self.bigram_embed.weight)
        self.n_layer = config.n_layer
        # Per-layer learnable scaling for bigram embedding injection
        self.bigram_lambdas = nn.Parameter(torch.full((config.n_layer,), 0.1))
        self._cached_bigram = None

    def _bigram_hash(self, idx):
        \"\"\"Compute bigram hash indices from consecutive token pairs.\"\"\"
        rand_int_1 = 36313
        rand_int_2 = 27191
        mod = self.bigram_vocab_size - 1
        x = idx.to(torch.int32)
        out = torch.zeros_like(x)
        # Position 0: no previous token, use reserved index
        out[:, 0] = mod
        # Positions 1+: XOR hash of (current, previous) token pair
        out[:, 1:] = torch.bitwise_xor(
            rand_int_1 * x[:, 1:],
            rand_int_2 * x[:, :-1]
        ) % mod
        return out.long()

    def forward(self, idx):
        b, t = idx.size()
        tok_emb = self.wte(idx)
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        pos_emb = self.wpe(pos)
        self._cached_bigram = self.bigram_embed(self._bigram_hash(idx))
        return self.drop(tok_emb + pos_emb)

    def get_value_embed(self, layer_idx):
        \"\"\"Inject bigram embedding at every layer with learnable scaling.\"\"\"
        if self._cached_bigram is None or layer_idx >= self.n_layer:
            return None
        return self.bigram_lambdas[layer_idx] * self._cached_bigram

    def get_lm_head_weight(self):
        return self.wte.weight

    def get_num_pos_params(self):
        return self.wpe.weight.numel()
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 115,
        "end_line": 140,
        "content": _BIGRAM_HASH_EMBEDDING,
    },
]
