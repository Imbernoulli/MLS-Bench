"""Value embeddings baseline (medium).

Adds learned value embeddings that are injected into selected transformer
layers as residual additions. Provides input-dependent value biases that
help the model attend to relevant token information at specific layers.

Reference: KellerJordan/modded-nanogpt records #14-15, #63
"""

_FILE = "nanoGPT/custom_pretrain.py"

_VALUE_EMBEDDING = """\
class TokenEmbedding(nn.Module):
    \"\"\"Token + position embedding with value embeddings for selected layers.\"\"\"
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.block_size = config.block_size
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        # Value embeddings: 5 tables injected into selected layers (like modded-nanogpt)
        self.n_ve = 5
        self.vte = nn.Embedding(config.vocab_size * self.n_ve, config.n_embd)
        nn.init.normal_(self.vte.weight, mean=0.0, std=0.01)
        # Per-VE learnable blending coefficient (lambda)
        self.ve_lambda = nn.Parameter(torch.full((self.n_ve,), 0.5))
        # Injection layers: layer 1, 2, and last 3 layers
        self._ve_layers = None
        self._cached_ve = None

    def forward(self, idx):
        b, t = idx.size()
        tok_emb = self.wte(idx)
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        pos_emb = self.wpe(pos)
        # Compute injection layer indices: layer 1, 2, and last 3 layers
        if self._ve_layers is None:
            self._ve_layers = [1, 2, self.n_layer - 3, self.n_layer - 2, self.n_layer - 1]
        # Cache per-VE value embeddings (5 separate lookups into partitioned table)
        vs = self.vocab_size
        self._cached_ve = {}
        for i, layer_idx in enumerate(self._ve_layers):
            offset_idx = idx + i * vs  # offset into partition i
            self._cached_ve[layer_idx] = self.vte(offset_idx)
        return self.drop(tok_emb + pos_emb)

    def get_value_embed(self, layer_idx):
        \"\"\"Get value embedding residual for a given layer, or None.\"\"\"
        if self._cached_ve is None or layer_idx not in self._cached_ve:
            return None
        ve_idx = self._ve_layers.index(layer_idx)
        lamb = self.ve_lambda[ve_idx]
        return lamb * self._cached_ve[layer_idx]

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
        "content": _VALUE_EMBEDDING,
    },
]
