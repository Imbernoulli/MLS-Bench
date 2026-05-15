"""Untied embeddings baseline (basic).

Separate the input token embedding from the output lm_head weight.
This adds parameters but allows the model to learn different representations
for input (contextual) and output (predictive) token distributions.

Contrast: Press & Wolf, "Using the Output Embedding to Improve Language
Models" (2017), which studies the tied-embedding alternative.
"""

_FILE = "nanoGPT/custom_pretrain.py"

_UNTIED_EMBEDDING = """\
class TokenEmbedding(nn.Module):
    \"\"\"Token + position embedding with UNTIED lm_head weight.\"\"\"
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.block_size = config.block_size
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size
        # Separate output projection weight (not tied to wte)
        self._lm_head_weight = nn.Parameter(torch.empty(config.vocab_size, config.n_embd))
        nn.init.zeros_(self._lm_head_weight)

    def forward(self, idx):
        b, t = idx.size()
        tok_emb = self.wte(idx)
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        pos_emb = self.wpe(pos)
        return self.drop(tok_emb + pos_emb)

    def get_lm_head_weight(self):
        return self._lm_head_weight

    def get_num_pos_params(self):
        return self.wpe.weight.numel()
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 115,
        "end_line": 140,
        "content": _UNTIED_EMBEDDING,
    },
]
