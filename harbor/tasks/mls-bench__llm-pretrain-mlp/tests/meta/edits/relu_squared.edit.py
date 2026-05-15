"""ReLU² (Squared ReLU) MLP baseline.

Replaces GELU with ReLU²: relu(x)^2. Simple but effective activation.
This task uses local validation rather than assuming a fixed gain over GELU.

Reference: So et al., "Primer: Searching for Efficient Transformers" (2021)
Inspired by modded-nanogpt record #5.
"""

_FILE = "nanoGPT/custom_pretrain.py"

_RELU_SQ_MLP = """\
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU²
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 72,
        "end_line": 86,
        "content": _RELU_SQ_MLP,
    },
]
