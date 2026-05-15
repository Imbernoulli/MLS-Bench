"""Cross-Attention baseline for ai4sci-weather-forecast-aggregation.

This is the default ClimaX aggregation mechanism: a learnable query token
attends to all variable tokens via multi-head cross-attention at each spatial
location independently.

Reference: vendor/external_packages/ClimaX/src/climax/arch.py (aggregate_variables)
Paper: Nguyen et al., "ClimaX: A Foundation Model for Weather and Climate", ICML 2023
"""

_FILE = "ClimaX/custom_forecast.py"

_CONTENT = """\
class VariableAggregator(nn.Module):
    \"\"\"Cross-attention variable aggregation (ClimaX default).

    A learnable query token attends to all V variable tokens at each spatial
    location via multi-head cross-attention, producing one token per location.

    Args:
        embed_dim (int): Embedding dimension D.
        num_heads (int): Number of attention heads.
        num_vars (int): Number of input variables V.
    \"\"\"

    def __init__(self, embed_dim, num_heads, num_vars):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_vars = num_vars
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        \"\"\"
        Args:
            x: [B, V, L, D] — per-variable patch embeddings.
        Returns:
            [B, L, D] — aggregated representation.
        \"\"\"
        b, v, l, d = x.shape
        x = x.permute(0, 2, 1, 3)   # B, L, V, D
        x = x.reshape(b * l, v, d)  # B*L, V, D

        query = self.var_query.expand(b * l, -1, -1)  # B*L, 1, D
        out, _ = self.var_agg(query, x, x)             # B*L, 1, D
        out = out.squeeze(1)                            # B*L, D

        out = out.reshape(b, l, d)  # B, L, D
        return out
"""

OPS = [
    {"op": "replace", "file": _FILE, "start_line": 310, "end_line": 351, "content": _CONTENT},
]
